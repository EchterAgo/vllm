# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/set_dynamic_sd_schedule")
async def set_dynamic_sd_schedule(raw_request: Request):
    """Hot-swap the dynamic speculative-decoding batch-size schedule.

    Body: ``{"num_speculative_tokens_per_batch_size": [[start, end, K], ...]}``
    or ``{"num_speculative_tokens_per_batch_size": null}`` to disable dynamic
    SD (fall back to the static max K). Requires an idle engine and that every
    K was captured at boot (present in the boot schedule's CUDA-graph tiers).
    Returns the resulting dense ``batch_size -> K`` lookup.
    """
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"JSON decode error: {e}",
        ) from e
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Request body must be a JSON object",
        )
    schedule = body.get("num_speculative_tokens_per_batch_size")
    if schedule is not None and not isinstance(schedule, list):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="'num_speculative_tokens_per_batch_size' must be a list or null",
        )
    try:
        lookup = await engine_client(raw_request).set_dynamic_sd_schedule(schedule)
    except Exception as e:  # noqa: BLE001 - core wraps utility failures generically
        msg = str(e)
        if "idle engine" in msg:
            raise HTTPException(
                status_code=HTTPStatus.CONFLICT.value, detail=msg
            ) from e
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=msg
        ) from e
    return JSONResponse(content={"dense_schedule": lookup})


@router.get("/get_dynamic_sd_schedule")
async def get_dynamic_sd_schedule(raw_request: Request):
    """Return the current dense ``batch_size -> K`` speculative schedule."""
    lookup = await engine_client(raw_request).get_dynamic_sd_schedule()
    return JSONResponse(content={"dense_schedule": lookup})


@router.post("/collective_rpc")
async def collective_rpc(raw_request: Request):
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"JSON decode error: {e}",
        ) from e
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Request body must be a JSON object",
        )
    method = body.get("method")
    if method is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'method' in request body",
        )
    # For security reason, only serialized string args/kwargs are passed.
    # User-defined `method` is responsible for deserialization if needed.
    args: list[str] = body.get("args", [])
    kwargs: dict[str, str] = body.get("kwargs", {})
    timeout: float | None = body.get("timeout")
    results = await engine_client(raw_request).collective_rpc(
        method=method, timeout=timeout, args=tuple(args), kwargs=kwargs
    )
    if results is None:
        return Response(status_code=200)
    response: list[Any] = []
    for result in results:
        if result is None or isinstance(result, dict | list):
            response.append(result)
        else:
            response.append(str(result))
    return JSONResponse(content={"results": response})


def attach_router(app: FastAPI):
    app.include_router(router)
