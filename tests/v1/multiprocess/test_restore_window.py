# SPDX-License-Identifier: Apache-2.0
"""Windowed restore of external prefixes larger than the L1 pin limit.

A loading lookup pins (loads and read-locks) at most ``restore_pin_limit_chunks``
chunks and reports the rest of the prefix from the L2 index. Workers then ask
the server for one window at a time (``RESTORE_WINDOW``), wait for it and
retrieve it. These tests cover the lookup module's side: the pin limit handed
to the storage manager, the recorded pinned prefix, lock releases that never
touch unpinned keys, and the window handler's job bookkeeping.
"""

# Standard
from unittest.mock import MagicMock
import threading
import time

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey, PrefetchHandle, PrefetchMode
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule, _PrefetchJob
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.multiprocess.protocols.engine import RestoreWindowResponse

CHUNK = 4


def _key(**overrides) -> IPCCacheServerKey:
    fields = dict(
        model_name="model",
        world_size=2,
        worker_id=None,
        token_ids=tuple(range(40)),
        start=0,
        end=40,
        request_id="req-1",
    )
    fields.update(overrides)
    return IPCCacheServerKey(**fields)


def _object_key(chunk: int, rank: int, group: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk),
        model_name="model",
        kv_rank=rank,
        object_group_id=group,
    )


def _handle(pinned_key_count: int, total: int, lookup_only: bool = False):
    return PrefetchHandle(
        prefetch_request_id=1,
        external_request_id="req-1",
        l1_found_indices=(),
        total_requested_keys=total,
        submit_time=time.monotonic(),
        lookup_only=lookup_only,
        pinned_key_count=pinned_key_count,
    )


def _module(pin_limit_chunks: int = 0) -> tuple[LookupModule, MagicMock]:
    ctx = MagicMock()
    ctx.chunk_size = CHUNK
    ctx.restore_pin_limit_chunks = pin_limit_chunks
    ctx.token_hasher.compute_chunk_hashes.side_effect = (
        lambda token_ids, prefix_hash=None, start=0, end=None: [
            ObjectKey.IntHash2Bytes(c)
            for c in range(start // CHUNK, (len(token_ids) if end is None else end) // CHUNK)
        ]
    )
    ctx.layout_desc_registry.find_object_group_layouts.return_value = [
        MagicMock(name="layout-0"),
        MagicMock(name="layout-1"),
    ]
    ctx.layout_desc_registry.find_attn_desc.return_value = MagicMock(
        num_object_groups=2, num_chunks_in_sw=[-1, -1]
    )
    module = object.__new__(LookupModule)
    module._ctx = ctx
    module._prefetch_jobs = {}
    module._prefetch_job_lock = threading.Lock()
    module._pinned_chunk_end = {}
    module._window_job_readers = {}
    return module, ctx


def test_restore_window_protocol_definition():
    assert get_payload_classes(RequestType.RESTORE_WINDOW) == [IPCCacheServerKey, int]
    assert get_response_class(RequestType.RESTORE_WINDOW) is RestoreWindowResponse


def test_lookup_hands_the_pin_limit_in_keys_to_the_storage_manager():
    module, ctx = _module(pin_limit_chunks=3)
    ctx.storage_manager.submit_prefetch_task.return_value = _handle(6, 20)

    module.lookup(_key(), tp_size=1)

    calls = ctx.storage_manager.submit_prefetch_task.call_args_list
    assert len(calls) == 2  # one task per object group
    for call in calls:
        # 3 chunks x world_size 2 ranks = 6 keys may be pinned per group.
        assert call.kwargs["pin_limit_keys"] == 6
        assert call.kwargs["mode"] is PrefetchMode.LOOKUP


def test_lookup_only_keys_never_pin():
    module, ctx = _module(pin_limit_chunks=3)
    ctx.storage_manager.submit_prefetch_task.return_value = _handle(-1, 20, True)

    module.lookup(_key(lookup_only=True), tp_size=1)

    for call in ctx.storage_manager.submit_prefetch_task.call_args_list:
        assert call.kwargs["pin_limit_keys"] == 0
        assert call.kwargs["mode"] is PrefetchMode.LOOKUP_ONLY


def _job_with_results(module, ctx, pinned_key_count, found_indices_by_group):
    keys_by_group = tuple(
        tuple(_object_key(c, r, g) for c in range(10) for r in range(2))
        for g in range(2)
    )
    module._prefetch_jobs["req-1"] = _PrefetchJob(
        handles=tuple(_handle(pinned_key_count, 20) for _ in range(2)),
        world_size=2,
        request_id="req-1",
        requested_tokens=40,
        object_keys_by_group=keys_by_group,
        extra_count=0,
    )
    bitmaps = []
    for indices in found_indices_by_group:
        bitmap = Bitmap(20)
        bitmap.batched_set(indices)
        bitmaps.append(bitmap)
    ctx.storage_manager.query_prefetch_status.side_effect = bitmaps
    return keys_by_group


def test_status_records_the_pinned_prefix_and_releases_only_locked_surplus():
    module, ctx = _module(pin_limit_chunks=3)
    # Both groups report 10 chunks (all 20 keys); the pin limit loaded 3
    # chunks = 6 keys per group, the other 14 keys are presence-only.
    keys = _job_with_results(module, ctx, 6, [list(range(20)), list(range(20))])

    assert module.query_prefetch_status("req-1") == 10
    assert module._pinned_chunk_end["req-1"] == 3
    # No surplus beyond the common prefix -> nothing released.
    ctx.storage_manager.finish_read_prefetched.assert_not_called()
    del keys


def test_status_releases_surplus_below_the_pinned_count_only():
    module, ctx = _module(pin_limit_chunks=3)
    # Group 0 found every key, group 1 only the first chunk (2 keys): the
    # common prefix is 1 chunk. Group 0's surplus keys 2..5 are locked (they
    # are within the 6 pinned keys) and get released; keys 6..19 are not.
    keys = _job_with_results(module, ctx, 6, [list(range(20)), [0, 1]])

    assert module.query_prefetch_status("req-1") == 1
    assert module._pinned_chunk_end["req-1"] == 1
    ctx.storage_manager.finish_read_prefetched.assert_called_once()
    released = ctx.storage_manager.finish_read_prefetched.call_args.args[0]
    assert released == list(keys[0][2:6])


def test_free_lookup_locks_is_clipped_to_the_pinned_prefix():
    module, ctx = _module(pin_limit_chunks=3)
    module._pinned_chunk_end["req-1"] = 3

    module.free_lookup_locks(_key(start=0, end=40), tp_size=1)

    args = ctx.token_hasher.compute_chunk_hashes.call_args
    assert args.kwargs["start"] == 0 and args.kwargs["end"] == 12
    released = ctx.storage_manager.finish_read_prefetched.call_args.args[0]
    assert len(released) == 3 * 2 * 2  # chunks x groups x ranks

    ctx.storage_manager.finish_read_prefetched.reset_mock()
    module.free_lookup_locks(_key(start=12, end=40), tp_size=1)
    ctx.storage_manager.finish_read_prefetched.assert_not_called()


def test_free_lookup_locks_without_a_record_releases_the_whole_range():
    module, ctx = _module(pin_limit_chunks=0)
    module.free_lookup_locks(_key(start=0, end=40), tp_size=1)
    released = ctx.storage_manager.finish_read_prefetched.call_args.args[0]
    assert len(released) == 10 * 2 * 2


def test_free_lookup_locks_for_a_worker_key_releases_the_exact_range():
    # A worker releases what it read-locked itself: a restore window it loaded
    # past the pinned prefix, or the pinned tail it will not retrieve. Neither
    # range is clipped to the pinned prefix, which bounds lookup keys only.
    module, ctx = _module(pin_limit_chunks=3)
    module._pinned_chunk_end["req-1"] = 3

    # Window [12, 28) = chunks 3..6, all past the pinned prefix.
    module.free_lookup_locks(_key(worker_id=1, start=12, end=28), tp_size=1)

    args = ctx.token_hasher.compute_chunk_hashes.call_args
    assert args.kwargs["start"] == 12 and args.kwargs["end"] == 28
    released = ctx.storage_manager.finish_read_prefetched.call_args.args[0]
    assert len(released) == 4 * 2  # chunks x groups, this rank only
    assert len({k.kv_rank for k in released}) == 1

    # The pinned tail [8, 12) = chunk 2 is released the same way.
    ctx.storage_manager.finish_read_prefetched.reset_mock()
    module.free_lookup_locks(_key(worker_id=1, start=8, end=12), tp_size=1)
    released = ctx.storage_manager.finish_read_prefetched.call_args.args[0]
    assert len(released) == 1 * 2


def test_restore_window_is_unknown_without_a_lookup_record():
    module, ctx = _module(pin_limit_chunks=3)
    response = module.restore_window(_key(worker_id=1, start=0, end=8), tp_size=1)
    assert response == RestoreWindowResponse(known=False)
    ctx.storage_manager.submit_prefetch_task.assert_not_called()


def test_restore_window_skips_the_pinned_prefix_and_loads_the_rest():
    module, ctx = _module(pin_limit_chunks=3)
    module._pinned_chunk_end["req-1"] = 3
    ctx.storage_manager.submit_prefetch_task.return_value = _handle(-1, 2)

    fully_pinned = module.restore_window(
        _key(worker_id=1, start=0, end=8), tp_size=1
    )
    assert fully_pinned == RestoreWindowResponse(
        known=True, pinned_chunk_end=3, submitted_chunks=0
    )
    ctx.storage_manager.submit_prefetch_task.assert_not_called()

    # Window [8, 20) = chunks 2..4; chunk 2 is pinned, chunks 3 and 4 load.
    response = module.restore_window(_key(worker_id=1, start=8, end=20), tp_size=1)
    assert response.known and response.pinned_chunk_end == 3
    assert response.submitted_chunks == 2
    assert response.job_id in module._prefetch_jobs
    calls = ctx.storage_manager.submit_prefetch_task.call_args_list
    assert len(calls) == 2  # one per object group
    for group, call in enumerate(calls):
        keys = call.args[0]
        assert [k.object_group_id for k in keys] == [group, group]
        assert [k.chunk_hash for k in keys] == [
            ObjectKey.IntHash2Bytes(3),
            ObjectKey.IntHash2Bytes(4),
        ]
        assert call.kwargs["mode"] is PrefetchMode.LOOKUP
        assert call.kwargs["external_request_id"] == response.job_id
    job = module._prefetch_jobs[response.job_id]
    assert job.is_window and job.world_size == 1
    assert job.requested_tokens == 2 * CHUNK


def test_restore_window_job_is_shared_by_its_readers():
    module, ctx = _module(pin_limit_chunks=0)
    module._pinned_chunk_end["req-1"] = 0
    ctx.storage_manager.submit_prefetch_task.return_value = _handle(-1, 2)
    key = _key(worker_id=0, start=0, end=8, readers_per_object=2)

    first = module.restore_window(key, tp_size=2)
    second = module.restore_window(key, tp_size=2)
    assert first == second
    assert ctx.storage_manager.submit_prefetch_task.call_count == 2  # groups, once
    assert module._window_job_readers[first.job_id] == 2

    loaded = Bitmap(2)
    loaded.batched_set([0, 1])
    ctx.storage_manager.query_prefetch_status.side_effect = [loaded, loaded]
    assert module.query_prefetch_status(first.job_id) == 2
    # The first reader leaves the job for the second one.
    assert first.job_id in module._prefetch_jobs
    assert module._window_job_readers[first.job_id] == 1
    assert module.query_prefetch_status(first.job_id) == 2
    assert first.job_id not in module._prefetch_jobs
    assert first.job_id not in module._window_job_readers
    # A window job records no pinned prefix of its own.
    assert first.job_id not in module._pinned_chunk_end


def test_end_session_forgets_the_pinned_prefix():
    module, ctx = _module(pin_limit_chunks=3)
    module._pinned_chunk_end["req-1"] = 3
    ctx.session_manager.remove.return_value = None
    module.end_session("req-1")
    assert "req-1" not in module._pinned_chunk_end
