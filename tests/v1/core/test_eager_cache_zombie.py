# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager prefix-cache registration zombie protection (upstream PR #43569).

``allocate_slots`` registers block hashes in the prefix cache before the
worker writes the KV bytes. If the request is preempted or aborted before
the worker runs, the hash entry survives as a "zombie": a later request
with the same prefix cache-hits a block that was never written. The
block pool tracks eager registrations in per-step buckets; the preempt/
abort paths roll them back before freeing, and worker-confirmed paths
(connector load completion, async-scheduler post-output) register as
``committed`` so a later rollback cannot evict valid entries.
"""

from __future__ import annotations

import pytest

from tests.v1.core.test_prefix_caching import (
    make_kv_cache_config,
    make_kv_cache_config_hybrid_model,
    make_request,
)
from tests.v1.core.utils import create_scheduler
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def _make_manager(kv_cache_config, **kwargs) -> KVCacheManager:
    kwargs.setdefault(
        "scheduler_block_size",
        max(g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups),
    )
    return KVCacheManager(
        kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=kwargs.pop("hash_block_size", 16),
        **kwargs,
    )


def _uncommitted_blocks_for(manager: KVCacheManager, req_id: str) -> list:
    """Flatten all eager buckets and return req_id's tracked blocks."""
    out: list = []
    for bucket in manager.block_pool._eager_uncommitted:
        out.extend(bucket.get(req_id, []))
    return out


def test_rollback_uncommitted_evicts_preempt_zombies():
    """Preempt cleanup (``rollback_uncommitted`` + ``free``) must evict the
    eager hash entries so a later request with the same prefix does NOT
    cache-hit never-written blocks."""
    block_size = 16
    manager = _make_manager(make_kv_cache_config(block_size, 11))

    # req_A: 32 tokens = 2 full blocks.
    common_token_ids = [i for i in range(2) for _ in range(block_size)]
    req_a = make_request("a", common_token_ids, block_size, sha256)
    computed_a, num_computed_a, _ = manager.get_computed_blocks(req_a)
    assert num_computed_a == 0  # cold cache
    manager.begin_eager_step()
    blocks_a = manager.allocate_slots(req_a, 32, 0, computed_a)
    assert blocks_a is not None

    cache_map = manager.block_pool.cached_block_hash_to_block._cache
    assert len(cache_map) == 2
    assert len(_uncommitted_blocks_for(manager, "a")) == 2

    block_ids_a = blocks_a.get_block_ids()[0]
    eager_blocks = [manager.block_pool.blocks[bid] for bid in block_ids_a]

    # Simulate preempt before the worker write.
    manager.rollback_uncommitted(req_a.request_id)
    manager.free(req_a)

    assert len(cache_map) == 0, (
        f"rollback must evict both eager hash entries, got {len(cache_map)}."
    )
    assert _uncommitted_blocks_for(manager, "a") == []
    for blk in eager_blocks:
        assert blk.ref_cnt == 0, "free() did not decrement ref_cnt"
        assert blk.block_hash is None, (
            "block.block_hash must be reset by the rollback path."
        )

    # req_B: superset prefix -> cache MISS.
    req_b_tokens = common_token_ids + [99] * block_size
    req_b = make_request("b", req_b_tokens, block_size, sha256)
    _, num_computed_b, _ = manager.get_computed_blocks(req_b)
    assert num_computed_b == 0, (
        f"req_b must not cache-hit on rolled-back entries; got "
        f"{num_computed_b} cached tokens."
    )


def test_rollback_uncommitted_covers_mamba_hybrid_groups():
    """Same race for full-attention + Mamba groups: the rollback path covers
    every manager that caches during ``allocate_slots``."""
    block_size = 16
    manager = _make_manager(
        make_kv_cache_config_hybrid_model(
            block_size,
            num_blocks=16,
            sliding_window_blocks=0,
            second_spec_type="mamba",
        )
    )

    common_token_ids = [i for i in range(2) for _ in range(block_size)]
    req_a = make_request("a", common_token_ids, block_size, sha256)
    computed_a, _, _ = manager.get_computed_blocks(req_a)
    manager.begin_eager_step()
    blocks_a = manager.allocate_slots(req_a, 32, 0, computed_a)
    assert blocks_a is not None

    cache_map = manager.block_pool.cached_block_hash_to_block._cache
    eager_entry_count = len(cache_map)
    assert eager_entry_count > 0
    assert len(_uncommitted_blocks_for(manager, "a")) == eager_entry_count

    manager.rollback_uncommitted(req_a.request_id)
    manager.free(req_a)

    assert len(cache_map) == 0, (
        f"rollback must clear all groups' eager entries; got {len(cache_map)}."
    )
    assert _uncommitted_blocks_for(manager, "a") == []

    req_b = make_request("b", common_token_ids, block_size, sha256)
    _, num_computed_b, _ = manager.get_computed_blocks(req_b)
    assert num_computed_b == 0


def test_finish_path_free_alone_preserves_cache_entries():
    """The normal-finish path (free after the step's commit) must keep cache
    entries hittable for future requests."""
    block_size = 16
    manager = _make_manager(make_kv_cache_config(block_size, 11))

    common_token_ids = [i for i in range(2) for _ in range(block_size)]
    req_a = make_request("a", common_token_ids, block_size, sha256)
    computed_a, _, _ = manager.get_computed_blocks(req_a)
    manager.begin_eager_step()
    manager.allocate_slots(req_a, 32, 0, computed_a)
    assert len(_uncommitted_blocks_for(manager, "a")) == 2

    # The step's worker confirmed its writes: commit retires the bucket, so a
    # later rollback (e.g. from the finish-path free) cannot evict these.
    manager.commit_eager_step()
    manager.free(req_a)

    cache_map = manager.block_pool.cached_block_hash_to_block._cache
    assert len(cache_map) == 2, (
        "Normal finish must leave the cache entries intact for future hits."
    )
    assert _uncommitted_blocks_for(manager, "a") == []

    # A later request with the same prefix hits the committed blocks (capped
    # at num_tokens - 1, so one full block of the two).
    req_b = make_request("b", common_token_ids, block_size, sha256)
    _, num_computed_b, _ = manager.get_computed_blocks(req_b)
    assert num_computed_b == block_size


def test_cache_blocks_committed_kwarg_skips_uncommitted_tracking():
    """``cache_blocks(committed=True)`` (connector load completion, async
    post-output) must not be tracked in the eager buckets, so a later
    rollback leaves those entries intact."""
    block_size = 16
    manager = _make_manager(make_kv_cache_config(block_size, 11))

    tokens = [i for i in range(2) for _ in range(block_size)]
    req = make_request("a", tokens, block_size, sha256)
    computed, _, _ = manager.get_computed_blocks(req)
    manager.begin_eager_step()
    blocks = manager.allocate_slots(req, 32, 0, computed, delay_cache_blocks=True)
    assert blocks is not None
    # Deferred: nothing cached yet.
    cache_map = manager.block_pool.cached_block_hash_to_block._cache
    assert len(cache_map) == 0

    # Worker-confirmed registration (e.g. connector load finished).
    manager.cache_blocks(req, 32, committed=True)
    assert len(cache_map) == 2
    assert _uncommitted_blocks_for(manager, "a") == []

    # A rollback for the same request must not evict committed entries.
    manager.rollback_uncommitted("a")
    assert len(cache_map) == 2


def test_reset_prefix_cache_clears_eager_buckets():
    """``reset_prefix_cache`` resets every block's hash; the eager buckets
    must be cleared alongside so a later rollback cannot walk stale entries
    pointing at re-allocated blocks."""
    block_size = 16
    manager = _make_manager(make_kv_cache_config(block_size, 11))

    tokens = [i for i in range(2) for _ in range(block_size)]
    req = make_request("a", tokens, block_size, sha256)
    computed, _, _ = manager.get_computed_blocks(req)
    manager.begin_eager_step()
    manager.allocate_slots(req, 32, 0, computed)
    assert len(_uncommitted_blocks_for(manager, "a")) == 2
    assert len(manager.block_pool._eager_uncommitted) == 1

    manager.free(req)
    assert manager.reset_prefix_cache()
    assert len(manager.block_pool.cached_block_hash_to_block._cache) == 0
    assert len(manager.block_pool._eager_uncommitted) == 0, (
        "reset_prefix_cache must clear the eager buckets."
    )


def test_scheduler_preempt_rolls_back_target_step_eager_cache():
    """Exercise the rollback through a real ``Scheduler.schedule()``.

    Two RUNNING chunked-prefill requests each have one committed block. In
    the target step req_A eagerly caches two more full blocks, then req_B
    cannot allocate, so priority preemption removes req_A from the same
    scheduler output before any worker sees it. The preempt path rolls back
    req_A's target-step eager entries; the prior-step seeded block survives,
    so a future req_C with req_A's first two blocks as prefix hits exactly
    the one seeded block.
    """
    block_size = 16
    # Budget forces the race: 2 free blocks cover req_a's target-step
    # allocation, leaving req_b at 0 -> preempt of the lowest-priority
    # running request (req_a).
    scheduler = create_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=48,
        max_model_len=2048,
        enable_prefix_caching=True,
        num_blocks=5,
        block_size=block_size,
        scheduling_policy="priority",
    )
    manager = scheduler.kv_cache_manager

    tokens_a = [0] * block_size + [1] * block_size + [2] * block_size
    tokens_b = [10] * block_size + [11] * block_size
    req_a = make_request("a", tokens_a, block_size, sha256)
    req_b = make_request("b", tokens_b, block_size, sha256)
    # Smaller value = higher priority; req_b outranks req_a.
    req_a.priority = 5
    req_b.priority = 0
    req_a.arrival_time = 1.0
    req_b.arrival_time = 2.0

    # Plant both requests in RUNNING with their first block already computed,
    # bypassing admission to start mid-chunked-prefill (the only state where
    # this race opens).
    for req in (req_a, req_b):
        seeded_blocks = manager.allocate_slots(req, block_size)
        assert seeded_blocks is not None
        req.num_computed_tokens = block_size
        req.status = RequestStatus.RUNNING
        scheduler.requests[req.request_id] = req
    scheduler.running = [req_a, req_b]

    # The seed allocations registered hashes in the current bucket; production
    # puts seed entries in a *prior* committed step. Commit to model that.
    manager.commit_eager_step()

    cache_map = manager.block_pool.cached_block_hash_to_block._cache
    assert len(cache_map) == 2  # one committed hash per seeded request
    assert manager.block_pool.get_num_free_blocks() == 2  # the race window

    output = scheduler.schedule()

    assert output.preempted_req_ids == {"a"}
    assert "a" not in output.num_scheduled_tokens, (
        "req_a was removed from the scheduler output before worker execution."
    )
    assert output.num_scheduled_tokens == {"b": block_size}
    assert req_a.status == RequestStatus.PREEMPTED

    # A future request with A's first two blocks as prefix hits exactly the
    # one seeded (committed) block; the target-step eager entry was rolled
    # back on preempt.
    req_c = make_request(
        "c",
        tokens_a[: 2 * block_size] + [99] * block_size,
        block_size,
        sha256,
    )
    computed_c, num_computed_c, _ = manager.get_computed_blocks(req_c)
    assert num_computed_c == block_size, (
        "Scheduler-level preemption must roll back the target-step eager "
        f"cache entry; got {num_computed_c} cached tokens, expected "
        f"{block_size}."
    )
    assert len(computed_c.blocks[0]) == 1
