# SPDX-License-Identifier: Apache-2.0
"""Lookup-only intent on the IPC lookup key and in the server lookup module."""

# Standard
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock
import time

# Third Party
import msgspec

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    ObjectKey,
    PrefetchMode,
)
from lmcache.v1.distributed.storage_manager import PrefetchHandle
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule, _PrefetchJob


def _make_key(**overrides) -> IPCCacheServerKey:
    fields = dict(
        model_name="test_model",
        world_size=2,
        worker_id=None,
        token_ids=tuple(range(8)),
        start=0,
        end=8,
        request_id="req-1",
        cache_salt="",
    )
    fields.update(overrides)
    return IPCCacheServerKey(**fields)


def test_lookup_only_defaults_to_false_and_is_not_cache_identity():
    key = _make_key()
    assert key.lookup_only is False
    marked = replace(key, lookup_only=True)
    assert marked.lookup_only is True
    assert marked == key
    assert marked.no_worker_id_version().lookup_only is True
    assert key.no_worker_id_version().lookup_only is False


def test_lookup_only_survives_msgspec_roundtrip():
    key = replace(_make_key(), lookup_only=True)
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(key), type=IPCCacheServerKey
    )
    assert decoded.lookup_only is True
    assert decoded == key


def test_payload_without_the_field_decodes_as_a_loading_lookup():
    key = _make_key()
    legacy_payload = {
        "model_name": key.model_name,
        "world_size": key.world_size,
        "worker_id": key.worker_id,
        "token_ids": list(key.token_ids),
        "start": key.start,
        "end": key.end,
        "request_id": key.request_id,
        "cache_salt": key.cache_salt,
        "readers_per_object": key.readers_per_object,
    }
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(legacy_payload), type=IPCCacheServerKey
    )
    assert decoded == key
    assert decoded.lookup_only is False


def _module_with_job(lookup_only: bool) -> tuple[LookupModule, str, MagicMock]:
    ctx = MagicMock()
    ctx.token_hasher.chunk_size = 4
    ctx.chunk_size = 4
    module = LookupModule(ctx)
    keys_a = tuple(
        ObjectKey(
            chunk_hash=ObjectKey.IntHash2Bytes(i),
            model_name="test_model",
            kv_rank=0,
            object_group_id=0,
        )
        for i in range(4)
    )
    keys_b = tuple(
        ObjectKey(
            chunk_hash=ObjectKey.IntHash2Bytes(i),
            model_name="test_model",
            kv_rank=0,
            object_group_id=1,
        )
        for i in range(4)
    )
    handles = tuple(
        PrefetchHandle(
            prefetch_request_id=index,
            external_request_id="req-0",
            l1_found_indices=(),
            total_requested_keys=4,
            submit_time=time.monotonic(),
            lookup_only=lookup_only,
        )
        for index in range(2)
    )
    request_id = "req-1"
    module._prefetch_jobs[request_id] = _PrefetchJob(
        handles=handles,
        world_size=1,
        request_id=request_id,
        requested_tokens=16,
        object_keys_by_group=(keys_a, keys_b),
        lookup_only=lookup_only,
    )
    full = Bitmap(4)
    full.batched_set([0, 1, 2, 3])
    short = Bitmap(4)
    short.batched_set([0, 1])
    ctx.storage_manager.query_prefetch_status.side_effect = [full, short]
    return module, request_id, ctx


def test_status_of_a_lookup_only_job_releases_nothing():
    module, request_id, ctx = _module_with_job(lookup_only=True)
    assert module.query_prefetch_status(request_id) == 2
    ctx.storage_manager.finish_read_prefetched.assert_not_called()
    assert request_id not in module._prefetch_jobs


def test_status_of_a_loading_job_releases_the_group_surplus():
    module, request_id, ctx = _module_with_job(lookup_only=False)
    assert module.query_prefetch_status(request_id) == 2
    ctx.storage_manager.finish_read_prefetched.assert_called_once()
    surplus = ctx.storage_manager.finish_read_prefetched.call_args.args[0]
    assert [k.object_group_id for k in surplus] == [0, 0]


def test_lookup_submits_lookup_only_mode_from_the_key():
    ctx = MagicMock()
    ctx.token_hasher.chunk_size = 4
    ctx.chunk_size = 4
    ctx.token_hasher.compute_chunk_hashes.return_value = [b"h0", b"h1"]
    layout = object()
    ctx.layout_desc_registry.find_object_group_layouts.return_value = [layout]
    ctx.layout_desc_registry.find_attn_desc.return_value = SimpleNamespace(
        num_object_groups=1, num_chunks_in_sw=[-1]
    )
    ctx.event_bus.has_subscribers.return_value = False
    module = LookupModule(ctx)

    key = replace(_make_key(world_size=1), lookup_only=True)
    module.lookup(key, tp_size=1)

    call = ctx.storage_manager.submit_prefetch_task.call_args
    assert call.kwargs["mode"] is PrefetchMode.LOOKUP_ONLY
    assert isinstance(call.kwargs["attn_desc"], AttnWindowDesc)
    assert module._prefetch_jobs[key.request_id].lookup_only is True

    ctx.storage_manager.submit_prefetch_task.reset_mock()
    module.lookup(_make_key(world_size=1, request_id="req-2"), tp_size=1)
    call = ctx.storage_manager.submit_prefetch_task.call_args
    assert call.kwargs["mode"] is PrefetchMode.LOOKUP
    assert module._prefetch_jobs["req-2"].lookup_only is False
