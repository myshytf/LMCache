# SPDX-License-Identifier: Apache-2.0
"""``PrefetchMode.LOOKUP_ONLY``: presence reports without loads or locks.

A lookup-only prefetch answers "how many leading keys exist in L1 or L2" for
a caller that will not retrieve them. It must not copy objects from L2 into
L1, must not read-lock L1 objects, and must leave no L2 lookup lock behind.
"""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchMode,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    L2AdaptersConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_manager import StorageManager

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


def _wait_for(predicate, timeout: float = 10.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _hits(sm: StorageManager, handle, timeout: float = 10.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = sm.query_prefetch_status(handle)
        if found is not None:
            return found.count_leading_ones()
        time.sleep(0.02)
    raise TimeoutError("prefetch status never became available")


def _key(chunk: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk),
        model_name="test_model",
        kv_rank=0,
    )


@pytest.fixture
def layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


@pytest.fixture
def storage_manager():
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=128 * 1024 * 1024,
                use_lazy=True,
                init_size_in_bytes=64 * 1024 * 1024,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(
            adapters=[MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)],
        ),
    )
    sm = StorageManager(config)
    yield sm
    sm.close()


def _write_and_persist(sm: StorageManager, keys: list[ObjectKey], layout) -> None:
    reserved = sm.reserve_write(keys, layout, mode="new")
    assert len(reserved) == len(keys)
    sm.finish_write(list(reserved.keys()))
    adapter = sm._l2_adapters[0]
    assert _wait_for(lambda: all(adapter.debug_has_key(k) for k in keys)), (
        "keys were not persisted to the mock L2 adapter"
    )
    # Let the store controller release its read locks before the caller
    # inspects lock state.
    time.sleep(0.05)


def test_lookup_only_reports_l2_prefix_without_loading(storage_manager, layout):
    sm = storage_manager
    keys = [_key(i) for i in range(5)]
    _write_and_persist(sm, keys, layout)
    sm.clear()
    used, _ = sm._l1_manager.get_memory_usage()
    assert used == 0

    handle = sm.submit_prefetch_task(keys, layout, mode=PrefetchMode.LOOKUP_ONLY)
    assert handle.lookup_only
    assert handle.l1_found_indices == ()
    assert _hits(sm, handle) == 5

    # Nothing was copied into L1 and no L2 lookup lock survives completion.
    used, _ = sm._l1_manager.get_memory_usage()
    assert used == 0
    assert sm._l1_manager.num_objects() == 0
    adapter = sm._l2_adapters[0]
    assert _wait_for(lambda: adapter.report_status()["locked_key_count"] == 0)


def test_lookup_only_counts_l1_prefix_without_locking(storage_manager, layout):
    sm = storage_manager
    keys = [_key(i) for i in range(5)]
    _write_and_persist(sm, keys, layout)

    handle = sm.submit_prefetch_task(keys, layout, mode=PrefetchMode.LOOKUP_ONLY)
    assert handle.lookup_only
    assert handle.l1_found_indices == tuple(range(5))
    assert handle.prefetch_request_id == -1
    assert _hits(sm, handle) == 5
    # No read lock was taken: every key stays evictable.
    assert all(sm._l1_manager.is_key_evictable(k) for k in keys)


def test_lookup_only_reports_l1_prefix_and_l2_continuation(storage_manager, layout):
    sm = storage_manager
    keys = [_key(i) for i in range(6)]
    _write_and_persist(sm, keys, layout)
    # Drop the tail from L1 only; it stays in L2.
    deleted, skipped = sm.delete_l1_keys(keys[3:])
    assert (deleted, skipped) == (3, 0)

    handle = sm.submit_prefetch_task(keys, layout, mode=PrefetchMode.LOOKUP_ONLY)
    assert handle.l1_found_indices == (0, 1, 2)
    assert handle.l2_orig_indices == (3, 4, 5)
    assert _hits(sm, handle) == 6
    assert sm._l1_manager.num_objects() == 3
    assert all(sm._l1_manager.is_key_evictable(k) for k in keys[:3])


def test_lookup_only_reports_zero_when_absent(storage_manager, layout):
    sm = storage_manager
    keys = [_key(100 + i) for i in range(4)]
    handle = sm.submit_prefetch_task(keys, layout, mode=PrefetchMode.LOOKUP_ONLY)
    assert _hits(sm, handle) == 0
    assert sm._l1_manager.num_objects() == 0


def test_lookup_only_is_gap_aware_like_a_prefix_prefetch(storage_manager, layout):
    sm = storage_manager
    keys = [_key(200 + i) for i in range(5)]
    _write_and_persist(sm, [keys[0], keys[1], keys[3], keys[4]], layout)
    sm.clear()

    handle = sm.submit_prefetch_task(keys, layout, mode=PrefetchMode.LOOKUP_ONLY)
    # Key 2 was never stored, so the contiguous prefix stops at two keys.
    assert _hits(sm, handle) == 2
    assert sm._l1_manager.num_objects() == 0
