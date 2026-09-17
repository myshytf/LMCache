# SPDX-License-Identifier: Apache-2.0
"""Prefetch through an L1 residency hole.

The storage manager counts only the leading run of readable L1 keys as L1
hits and forwards every later key to the L2 prefetch, including keys that are
still resident in L1 (per-key LRU eviction and per-worker lock release leave
chunks partially resident). The prefetch controller must treat such resident
keys as available instead of failing their write reservation, otherwise the
reserved prefix breaks at the first resident key, every later key is reported
as a failed load and the request recomputes a prefix that both tiers hold.
"""

# Standard
import logging
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


def _wait_for(predicate, timeout: float = 10.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _found(sm: StorageManager, handle, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = sm.query_prefetch_status(handle)
        if found is not None:
            return found
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
                use_lazy=False,
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
    time.sleep(0.05)


def _l1_state(sm: StorageManager, key: ObjectKey):
    return sm._l1_manager.get_object_state(key)


def _read_locked(sm: StorageManager, key: ObjectKey) -> bool:
    state = _l1_state(sm, key)
    return state is not None and state.read_lock.is_locked()


@pytest.mark.parametrize("extra_count", [0, 2])
def test_resident_keys_after_an_l1_miss_do_not_break_the_prefix(
    storage_manager, layout, caplog, extra_count
):
    """Keys 2-3 stay resident while 0-1 and 4-5 were evicted: the whole
    prefix is served (0-1 and 4-5 loaded from L2, 2-3 read-locked in place)."""
    sm = storage_manager
    keys = [_key(i) for i in range(6)]
    _write_and_persist(sm, keys, layout)
    sm.delete_l1_keys(keys[:2] + keys[4:])
    assert all(_l1_state(sm, k) is not None for k in keys[2:4])
    assert all(_l1_state(sm, k) is None for k in keys[:2] + keys[4:])

    with caplog.at_level(logging.WARNING):
        handle = sm.submit_prefetch_task(
            keys, layout, extra_count=extra_count, mode=PrefetchMode.LOOKUP
        )
        assert handle.l1_found_indices == ()
        assert sm.wait_prefetch_status(handle, timeout=10.0)
        found = _found(sm, handle)

    assert found.count_leading_ones() == 6
    assert "failed to load" not in caplog.text
    for key in keys:
        assert _read_locked(sm, key), f"{key} is not read-locked for the reader"
    # Every key carries 1 + extra_count read locks: releasing them once with the
    # same extra_count leaves nothing locked. Loaded keys are temporary and go
    # away with their last reader; the resident keys stay and become evictable.
    sm.finish_read_prefetched(keys, extra_count=extra_count)
    assert not any(_read_locked(sm, k) for k in keys)
    assert all(sm._l1_manager.is_key_evictable(k) for k in keys[2:4])


def test_prefix_still_stops_at_a_key_missing_from_both_tiers(storage_manager, layout):
    sm = storage_manager
    keys = [_key(i) for i in range(6)]
    _write_and_persist(sm, keys[:3] + keys[4:], layout)  # key 3 never stored
    sm.delete_l1_keys(keys[:2] + keys[4:])  # key 2 stays resident

    handle = sm.submit_prefetch_task(keys, layout, mode=PrefetchMode.LOOKUP)
    assert sm.wait_prefetch_status(handle, timeout=10.0)
    found = _found(sm, handle)

    assert found.count_leading_ones() == 3
    assert all(_read_locked(sm, k) for k in keys[:3])
    # Nothing past the gap keeps a lock.
    assert not _read_locked(sm, keys[4])
    sm.finish_read_prefetched(keys[:3])
    assert not any(_read_locked(sm, k) for k in keys)
    assert sm._l1_manager.is_key_evictable(keys[2])


def test_all_keys_resident_after_a_leading_miss(storage_manager, layout):
    """Only key 0 was evicted; the L2 prefetch loads it and the rest is served
    from L1 without any load."""
    sm = storage_manager
    keys = [_key(i) for i in range(4)]
    _write_and_persist(sm, keys, layout)
    sm.delete_l1_keys(keys[:1])

    handle = sm.submit_prefetch_task(keys, layout, mode=PrefetchMode.LOOKUP)
    assert sm.wait_prefetch_status(handle, timeout=10.0)
    assert _found(sm, handle).count_leading_ones() == 4
    assert all(_read_locked(sm, k) for k in keys)
    sm.finish_read_prefetched(keys)
    assert not any(_read_locked(sm, k) for k in keys)
    assert all(sm._l1_manager.is_key_evictable(k) for k in keys[1:])


def test_warm_prefetch_keeps_resident_keys_unlocked(storage_manager, layout):
    sm = storage_manager
    keys = [_key(i) for i in range(4)]
    _write_and_persist(sm, keys, layout)
    sm.delete_l1_keys(keys[:1] + keys[2:3])  # keys 1 and 3 stay resident

    handle = sm.submit_prefetch_task(keys, layout, mode=PrefetchMode.WARM)
    assert sm.wait_prefetch_status(handle, timeout=10.0)
    found = _found(sm, handle)
    assert found.popcount() == 4
    # WARM pins nothing: resident and loaded keys alike are free afterwards.
    assert _wait_for(lambda: all(sm._l1_manager.is_key_evictable(k) for k in keys))
