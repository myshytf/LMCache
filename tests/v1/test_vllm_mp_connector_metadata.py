# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``LMCacheMPRequestMetadata.GetRetrieveMetadata``.

The retrieve op must only be emitted when the tracker's allocated block ids
actually cover the retrieve token range in every engine group:
``slice_block_ids_per_group`` validates chunk alignment only, and its plain
Python slice truncates silently, so an uncovered range would otherwise emit
an op with too few block ids.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
    _recurrent_safe_lookup_end,
)

CHUNK_TOKENS = 64


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (0, 0),
        (1, 0),
        (63, 0),
        (64, 0),
        (65, 64),
        (128, 64),
        (129, 128),
    ],
)
def test_recurrent_lookup_excludes_final_prompt_token(
    num_tokens: int, expected: int
) -> None:
    """A recurrent restore ends before the token vLLM recomputes for logits."""
    assert _recurrent_safe_lookup_end(num_tokens, CHUNK_TOKENS) == expected


def test_recurrent_lookup_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_tokens must be positive"):
        _recurrent_safe_lookup_end(128, 0)


def _tracker(
    num_tokens: int,
    lmcache_hit_tokens: int,
    vllm_hit_tokens: int,
    allocated_block_ids: dict[int, list[int]],
) -> LMCacheMPRequestTracker:
    """Build a tracker that is ready for retrieving over *num_tokens*."""
    request = SimpleNamespace(
        request_id="req-1",
        cache_salt="",
        all_token_ids=list(range(num_tokens)),
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.num_lmcache_hit_tokens = lmcache_hit_tokens
    tracker.num_vllm_hit_tokens = vllm_hit_tokens
    tracker.allocated_block_ids = allocated_block_ids
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD
    return tracker


@pytest.mark.parametrize(
    ("group_tokens_per_block", "allocated_block_ids", "expected_block_ids"),
    [
        # Real pages start at 1; block 0 denotes an absent recurrent checkpoint.
        # Single group, plain geometry: 64 tokens / 16 tokens per block.
        ([16], {0: [1, 2, 3, 4]}, [[1, 2, 3, 4]]),
        # Single group, DCP-scaled: one manager block id covers 64 tokens.
        ([64], {0: [5]}, [[5]]),
        # Hybrid geometries: each group sliced by its own tokens-per-block.
        ([16, 32], {0: [1, 2, 3, 4], 1: [10, 11]}, [[1, 2, 3, 4], [10, 11]]),
        # Extra allocated blocks beyond the range are fine (and not emitted).
        ([16], {0: [1, 2, 3, 4, 5, 6]}, [[1, 2, 3, 4]]),
    ],
)
def test_retrieve_metadata_emitted_when_allocation_covers_range(
    group_tokens_per_block: list[int],
    allocated_block_ids: dict[int, list[int]],
    expected_block_ids: list[list[int]],
) -> None:
    tracker = _tracker(
        num_tokens=CHUNK_TOKENS,
        lmcache_hit_tokens=CHUNK_TOKENS,
        vllm_hit_tokens=0,
        allocated_block_ids=allocated_block_ids,
    )

    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, CHUNK_TOKENS, group_tokens_per_block
    )

    assert metadata is not None
    assert metadata.direction == "RETRIEVE"
    assert metadata.op.start == 0
    assert metadata.op.end == CHUNK_TOKENS
    assert metadata.op.block_ids == expected_block_ids


@pytest.mark.parametrize(
    ("group_tokens_per_block", "allocated_block_ids"),
    [
        # One block short in the only group.
        ([16], {0: [0, 1, 2]}),
        # DCP-scaled group with no allocation at all.
        ([64], {0: []}),
        # Group 0 covered, group 1 one block short.
        ([16, 32], {0: [0, 1, 2, 3], 1: [10]}),
        # Group key missing entirely (counts as zero blocks).
        ([16, 16], {0: [0, 1, 2, 3]}),
    ],
)
def test_retrieve_metadata_suppressed_when_allocation_short(
    group_tokens_per_block: list[int],
    allocated_block_ids: dict[int, list[int]],
) -> None:
    """Any engine group whose allocation does not cover the retrieve range
    suppresses the op entirely: the request falls back to recompute instead
    of retrieving over silently truncated block-id lists."""
    tracker = _tracker(
        num_tokens=CHUNK_TOKENS,
        lmcache_hit_tokens=CHUNK_TOKENS,
        vllm_hit_tokens=0,
        allocated_block_ids=allocated_block_ids,
    )

    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, CHUNK_TOKENS, group_tokens_per_block
    )

    assert metadata is None


def test_retrieve_metadata_skip_tokens_preserved_on_emitted_op() -> None:
    """vLLM-hit tokens inside the first chunk round the start down and are
    carried as skip_first_n_tokens; coverage is still checked against the
    chunk-aligned end."""
    tracker = _tracker(
        num_tokens=CHUNK_TOKENS,
        lmcache_hit_tokens=CHUNK_TOKENS,
        vllm_hit_tokens=16,
        allocated_block_ids={0: [0, 1, 2, 3]},
    )

    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(tracker, CHUNK_TOKENS, [16])

    assert metadata is not None
    assert metadata.op.start == 0
    assert metadata.op.end == CHUNK_TOKENS
    assert metadata.op.skip_first_n_tokens == 16
