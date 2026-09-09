# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/offloader.py
"""Base classes for model parameter offloading."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING

import torch.nn as nn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.config import OffloadConfig

logger = init_logger(__name__)


def should_pin_memory() -> bool:
    """Check if pinned memory should be used for weight offloading.

    Combines the platform capability check with the user override env var.
    On unified-memory systems (e.g. GH200) pinned memory eats into GPU
    memory, so users can disable it via VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY.
    """
    return (
        is_pin_memory_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY
    )


# Anonymous mmaps backing exact-size pinned tensors, held for the process
# lifetime.  See pin_exact() -- these back model weights, which live as long as
# the model does; releasing them early un-maps memory CUDA still has registered.
_PINNED_MAPPINGS: dict[int, object] = {}


def pin_exact(t: "torch.Tensor") -> "torch.Tensor":
    """Return a page-locked copy of ``t`` occupying EXACTLY its own size.

    ``Tensor.pin_memory()`` allocates through torch's ``CachingHostAllocator``,
    which rounds every block up to the next power of two.  For the large,
    awkwardly-sized tensors weight offloading deals with that is not a rounding
    error but a multiplier: a 2.250 GiB expert block costs 4.008 GiB of pinned
    host memory (measured 1.78x, torch 2.13.0+cu130).  Offloading a MoE's experts
    therefore needs ~1.78x the host RAM the weights occupy and the process is
    OOM-killed during construction, long before ``cpu_offload_gb`` is reached.
    (For reference, llama.cpp pins the same class of weights at ~1.0x.)

    An anonymous mmap is page-aligned by construction and exactly sized;
    registering it with ``cudaHostRegister`` yields page-locked memory with no
    bucketing.  Falls back to ``pin_memory()`` if registration is unavailable.
    """
    import mmap

    import torch

    nbytes = t.numel() * t.element_size()
    if nbytes == 0 or not t.is_contiguous():
        # Non-contiguous inputs would need the strided layout rebuilt over the
        # mapping; not worth it -- these are small next to the expert blocks.
        return t.pin_memory()

    buf = None
    try:
        buf = mmap.mmap(-1, nbytes)
        # Build the tensor at its final dtype/shape directly -- no uint8->dtype
        # ->shape view chain, whose intermediate bases confuse downstream code
        # that inspects storage_offset()/is_contiguous().
        out = torch.frombuffer(buf, dtype=t.dtype, count=t.numel()).view(t.shape)
        # Register the WHOLE mapping (mmap rounds up to a page), not just nbytes.
        rc = torch.cuda.cudart().cudaHostRegister(out.data_ptr(), len(buf), 0)
        if int(rc) != 0:
            raise RuntimeError(f"cudaHostRegister returned {rc}")
        out.copy_(t)
        # ⚠️ LIFETIME: the mapping must outlive every view of it.  Attaching the
        # mmap to the returned tensor is NOT enough -- callers reassign
        # ``param.data`` and drop this object, the mmap is munmapped while still
        # CUDA-registered, and the next copy fails with cudaErrorInvalidValue.
        # Offloaded weights live for the life of the model, so hold a process-
        # level reference keyed by pointer.
        _PINNED_MAPPINGS[out.data_ptr()] = buf
        return out
    except Exception as e:  # pragma: no cover - platform fallback
        if buf is not None:
            try:
                buf.close()
            except Exception:
                pass
        logger.warning_once(
            "pin_exact() unavailable (%s); falling back to Tensor.pin_memory(), "
            "which rounds allocations up to the next power of two.", e
        )
        return t.pin_memory()


def release_pinned(ptr: int) -> bool:
    """Unregister and unmap an exact-size pinned buffer created by pin_exact().

    ⚠️ Callers that REPLACE an offloaded parameter must call this with the old
    ``data_ptr()``.  ``process_weights_after_loading`` repacks expert weights
    (e.g. for Marlin), so the same logical parameter is offloaded twice; without
    releasing the first mapping both copies are held and the process needs
    roughly double the host RAM.  ``Tensor.pin_memory()`` gets this for free --
    its caching allocator recycles the block when the old tensor dies -- so the
    exact-size path has to do the recycling explicitly.
    """
    import torch

    buf = _PINNED_MAPPINGS.pop(ptr, None)
    if buf is None:
        return False
    try:
        torch.cuda.cudart().cudaHostUnregister(ptr)
    except Exception:  # pragma: no cover
        pass
    try:
        buf.close()
    except Exception:  # pragma: no cover
        pass
    return True



"""
class relation:

BaseOffloader (ABC)
  * implemented by: UVAOffloader
  * implemented by: PrefetchOffloader
    * uses: _ModuleOffloader
        * uses: _BaseParamOffloader (ABC)
            * implemented by: _CpuParamOffloader
"""


class BaseOffloader(ABC):
    """Base class for model parameter offloading strategies.

    Offloaders control how model parameters are stored and loaded during
    inference. Different strategies trade memory for compute/transfer time.
    """

    supports_tower_offload: bool = False
    """Whether `wrap_modules` also accepts modules routed by
    `SupportsMultiModal._mark_tower_model`, outside the `make_layers` call.

    Offloaders whose `wrap_modules` may only be called on the decoder layer
    stack (e.g. `PrefetchOffloader`, which schedules prefetches over a
    circular layer stack) must keep this `False`.
    """

    @abstractmethod
    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
        prefix: str = "",
    ) -> list[nn.Module]:
        """Wrap modules with offloading logic.

        Args:
            modules_generator: Generator yielding modules to potentially offload.
            prefix: Name prefix prepended to parameter names before matching
                them against the offloading parameter set. Used when the
                modules are not the full model, so that name segments stay
                fully qualified (e.g. `visual` for a tower module).

        Returns:
            List of modules, potentially with offloading hooks installed.
        """
        pass

    def post_init(self):
        """Called after model construction completes.

        Offloaders can use this to:
        - Finalize parameter storage
        - Start initial prefetching
        - Allocate shared resources
        """
        return

    def sync_prev_onload(self) -> None:  # noqa: B027
        """Sync previous onload operations. Override in subclasses."""
        pass

    def join_after_forward(self) -> None:  # noqa: B027
        """Join streams after forward. Override in subclasses."""
        pass

    def _wait_for_layer(self, layer_idx: int) -> None:  # noqa: B027
        """Wait for layer prefetch. Override in subclasses."""
        pass

    def _start_prefetch(self, layer_idx: int) -> None:  # noqa: B027
        """Start layer prefetch. Override in subclasses."""
        pass


class NoopOffloader(BaseOffloader):
    """No-op offloader that returns modules as-is without any offloading."""

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
        prefix: str = "",
    ) -> list[nn.Module]:
        """Return modules unchanged."""
        return list(modules_generator)


# Global singleton offloader instance (defaults to no-op).
_instance: BaseOffloader = NoopOffloader()


def get_offloader() -> BaseOffloader:
    """Get the global offloader instance."""
    return _instance


def set_offloader(instance: BaseOffloader) -> None:
    """Set the global offloader instance."""
    global _instance
    _instance = instance
    if isinstance(instance, NoopOffloader):
        logger.debug_once("Offloader set to NoopOffloader (no offloading).")
    else:
        logger.info_once("Offloader set to %s", type(instance).__name__)


def create_offloader(offload_config: "OffloadConfig") -> BaseOffloader:
    """Create an offloader based on the offload configuration.

    Uses the explicit ``offload_backend`` selector.  When set to ``"auto"``,
    selects prefetch if ``offload_group_size > 0``, UVA if
    ``cpu_offload_gb > 0``, otherwise noop.
    """
    from vllm.model_executor.offloader.prefetch import PrefetchOffloader
    from vllm.model_executor.offloader.uva import UVAOffloader

    backend = offload_config.offload_backend
    uva = offload_config.uva
    prefetch = offload_config.prefetch

    if backend == "auto":
        if prefetch.offload_group_size > 0:
            backend = "prefetch"
        elif uva.cpu_offload_gb > 0:
            backend = "uva"
        else:
            return NoopOffloader()

    if backend == "prefetch":
        return PrefetchOffloader(
            group_size=prefetch.offload_group_size,
            num_in_group=prefetch.offload_num_in_group,
            prefetch_step=prefetch.offload_prefetch_step,
            offload_params=prefetch.offload_params,
            mode="cpu",
        )
    elif backend == "uva":
        return UVAOffloader(
            cpu_offload_max_bytes=int(uva.cpu_offload_gb * 1024**3),
            cpu_offload_params=uva.cpu_offload_params,
        )
    else:
        return NoopOffloader()
