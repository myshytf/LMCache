# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (
    LMCacheMPRequestMetadata,
    LMCacheMPRequestTracker,
)


def test_store_metadata_uses_external_chunk_geometry_for_dcp_groups() -> None:
    """Store metadata emits one block destination per fine external chunk."""
    tracker = object.__new__(LMCacheMPRequestTracker)
    tracker.__dict__.update(
        request_id="req",
        cache_salt="",
        all_token_ids=list(range(44_609)),
        allocated_block_ids={
            0: [10, 11, 12, 13],
            1: list(range(30)),
        },
        num_scheduled_tokens=43_008,
        num_vllm_hit_tokens=0,
        num_lmcache_hit_tokens=0,
        num_stored_tokens=36_864,
    )

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=1_536,
        group_tokens_per_block=[12_288, 1_536],
    )

    assert metadata is not None
    assert metadata.op.start == 36_864
    assert metadata.op.end == 43_008
    assert metadata.op.block_ids == [[13, 13, 13, 13], [24, 25, 26, 27]]
