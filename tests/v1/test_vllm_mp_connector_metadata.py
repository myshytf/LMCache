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
    LMCacheMPConnector,
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
        # Single group, plain geometry: 64 tokens / 16 tokens per block.
        # Block id 0 is vLLM's null placeholder, so real ids start at 1.
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
        allocated_block_ids={0: [1, 2, 3, 4]},
    )

    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(tracker, CHUNK_TOKENS, [16])

    assert metadata is not None
    assert metadata.op.start == 0
    assert metadata.op.end == CHUNK_TOKENS
    assert metadata.op.skip_first_n_tokens == 16


# ---------------------------------------------------------------------------
# Retrieve range bounded by the scheduler's admitted external tokens.
# ---------------------------------------------------------------------------


def _lookup_tracker(
    lmcache_hit_tokens: int,
    vllm_hit_tokens: int,
) -> LMCacheMPRequestTracker:
    """Build a tracker as it looks right after ``get_num_new_matched_tokens``."""
    tracker = _tracker(
        num_tokens=lmcache_hit_tokens + CHUNK_TOKENS,
        lmcache_hit_tokens=lmcache_hit_tokens,
        vllm_hit_tokens=vllm_hit_tokens,
        allocated_block_ids={0: list(range(1, 13))},
    )
    tracker.state = LMCacheMPRequestState.PREFETCHING
    return tracker


def test_retrieve_end_defaults_to_lookup_hit_before_admission() -> None:
    tracker = _lookup_tracker(3 * CHUNK_TOKENS, 0)

    assert tracker.retrieve_end_token(CHUNK_TOKENS) == 3 * CHUNK_TOKENS
    assert tracker.needs_retrieve()


@pytest.mark.parametrize(
    ("vllm_hit", "admitted", "expected_end", "expected_needs_retrieve"),
    [
        # Scheduler kept a local sub-block tail and admitted nothing.
        (0, 0, 0, False),
        # Full lookup hit admitted.
        (0, 3 * CHUNK_TOKENS, 3 * CHUNK_TOKENS, True),
        # Partial admission on an object boundary.
        (0, CHUNK_TOKENS, CHUNK_TOKENS, True),
        # Non-aligned admission rounds up to the object boundary.
        (0, CHUNK_TOKENS + 1, 2 * CHUNK_TOKENS, True),
        # Admission beyond the lookup hit is capped at the hit.
        (0, 5 * CHUNK_TOKENS, 3 * CHUNK_TOKENS, True),
        # Local hit plus admitted continuation.
        (CHUNK_TOKENS, CHUNK_TOKENS, 2 * CHUNK_TOKENS, True),
        (CHUNK_TOKENS, 0, CHUNK_TOKENS, False),
    ],
)
def test_retrieve_end_bounded_by_admitted_external_tokens(
    vllm_hit: int,
    admitted: int,
    expected_end: int,
    expected_needs_retrieve: bool,
) -> None:
    tracker = _lookup_tracker(3 * CHUNK_TOKENS, vllm_hit)

    tracker.admit_external_tokens(admitted)

    assert tracker.retrieve_end_token(CHUNK_TOKENS) == expected_end
    assert tracker.needs_retrieve() is expected_needs_retrieve


def test_retrieve_metadata_stops_at_admitted_external_range() -> None:
    tracker = _lookup_tracker(3 * CHUNK_TOKENS, 0)
    tracker.admit_external_tokens(2 * CHUNK_TOKENS)
    tracker.state = LMCacheMPRequestState.WAITING_FOR_LOAD

    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(tracker, CHUNK_TOKENS, [16])

    assert metadata is not None
    assert metadata.op.start == 0
    assert metadata.op.end == 2 * CHUNK_TOKENS
    assert metadata.op.block_ids == [list(range(1, 9))]


class _RecordingSchedulerAdapter:
    """Scheduler adapter stub recording lock releases."""

    lmcache_tokens_per_chunk = CHUNK_TOKENS

    def __init__(self) -> None:
        self.freed: list[tuple[int, int]] = []
        self.cleaned: list[str] = []

    def cleanup_lookup_result(self, request_id: str) -> None:
        self.cleaned.append(request_id)

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
    ) -> None:
        self.freed.append((start, end))


def _connector_with_tracker(
    tracker: LMCacheMPRequestTracker,
) -> tuple[LMCacheMPConnector, _RecordingSchedulerAdapter]:
    connector = object.__new__(LMCacheMPConnector)
    adapter = _RecordingSchedulerAdapter()
    connector.request_trackers = {tracker.request_id: tracker}
    connector.scheduler_adapter = adapter
    return connector, adapter


def test_update_state_after_alloc_with_zero_admitted_tokens_skips_retrieve() -> None:
    """A local sub-block tail kept by the scheduler must not trigger a
    retrieve that races the forward pass on the same blocks; every lookup
    lock is released instead."""
    tracker = _lookup_tracker(3 * CHUNK_TOKENS, 0)
    connector, adapter = _connector_with_tracker(tracker)
    request = SimpleNamespace(request_id=tracker.request_id)
    blocks = SimpleNamespace(get_block_ids=lambda: ([1, 2, 3, 4, 5, 6, 7, 8],))

    connector.update_state_after_alloc(request, blocks, 0)

    assert tracker.state is LMCacheMPRequestState.READY
    assert not tracker.needs_retrieve()
    assert adapter.freed == [(0, 3 * CHUNK_TOKENS)]
    assert adapter.cleaned == [tracker.request_id]


def test_update_state_after_alloc_partial_admission_frees_unadmitted_tail() -> None:
    tracker = _lookup_tracker(3 * CHUNK_TOKENS, 0)
    connector, adapter = _connector_with_tracker(tracker)
    request = SimpleNamespace(request_id=tracker.request_id)
    blocks = SimpleNamespace(get_block_ids=lambda: ([1, 2, 3, 4, 5, 6, 7, 8],))

    connector.update_state_after_alloc(request, blocks, CHUNK_TOKENS)

    assert tracker.state is LMCacheMPRequestState.WAITING_FOR_LOAD
    assert tracker.retrieve_end_token(CHUNK_TOKENS) == CHUNK_TOKENS
    assert adapter.freed == [(CHUNK_TOKENS, 3 * CHUNK_TOKENS)]


def test_update_state_after_alloc_full_admission_keeps_full_retrieve() -> None:
    tracker = _lookup_tracker(3 * CHUNK_TOKENS, 0)
    connector, adapter = _connector_with_tracker(tracker)
    request = SimpleNamespace(request_id=tracker.request_id)
    blocks = SimpleNamespace(get_block_ids=lambda: ([1, 2, 3, 4, 5, 6, 7, 8],))

    connector.update_state_after_alloc(request, blocks, 3 * CHUNK_TOKENS)

    assert tracker.state is LMCacheMPRequestState.WAITING_FOR_LOAD
    assert tracker.retrieve_end_token(CHUNK_TOKENS) == 3 * CHUNK_TOKENS
    assert adapter.freed == []


# ---------------------------------------------------------------------------
# Null placeholder blocks are never transfer sources or destinations.
# ---------------------------------------------------------------------------

HYBRID_GROUP_TOKENS = [8 * CHUNK_TOKENS, CHUNK_TOKENS]


def _hybrid_store_tracker(
    recurrent_block_ids: list[int],
    num_computed_tokens: int,
) -> LMCacheMPRequestTracker:
    """A hybrid request whose attention group owns one DCP-scaled block."""
    request = SimpleNamespace(
        request_id="req-hybrid",
        cache_salt="",
        all_token_ids=list(range(num_computed_tokens + 8)),
    )
    tracker = LMCacheMPRequestTracker(request)
    tracker.allocated_block_ids = {0: [13], 1: recurrent_block_ids}
    tracker.num_scheduled_tokens = num_computed_tokens
    return tracker


def test_store_stops_before_first_null_backed_chunk() -> None:
    """A recurrent checkpoint slot holding the null block has no state to
    persist; storing it would publish placeholder bytes as cache content."""
    tracker = _hybrid_store_tracker([1, 0, 3, 4], 4 * CHUNK_TOKENS)

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, CHUNK_TOKENS, HYBRID_GROUP_TOKENS
    )

    assert metadata is not None
    assert metadata.op.start == 0
    assert metadata.op.end == CHUNK_TOKENS
    assert metadata.op.block_ids == [[13], [1]]
    assert tracker.num_stored_tokens == CHUNK_TOKENS


def test_store_emits_nothing_when_first_chunk_is_null_backed() -> None:
    tracker = _hybrid_store_tracker([0, 0, 3, 4], 4 * CHUNK_TOKENS)

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, CHUNK_TOKENS, HYBRID_GROUP_TOKENS
    )

    assert metadata is None
    assert tracker.num_stored_tokens == 0


def test_store_keeps_full_range_without_null_blocks() -> None:
    tracker = _hybrid_store_tracker([1, 2, 3, 4], 4 * CHUNK_TOKENS)

    metadata = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker, CHUNK_TOKENS, HYBRID_GROUP_TOKENS
    )

    assert metadata is not None
    assert metadata.op.end == 4 * CHUNK_TOKENS
    assert metadata.op.block_ids == [[13, 13, 13, 13], [1, 2, 3, 4]]


def test_retrieve_marks_null_destinations_skipped() -> None:
    """Placeholder destinations reach the worker as skipped ids; the
    attention group's ids are untouched and coverage is still satisfied."""
    tracker = _tracker(
        num_tokens=3 * CHUNK_TOKENS,
        lmcache_hit_tokens=3 * CHUNK_TOKENS,
        vllm_hit_tokens=0,
        allocated_block_ids={0: [13], 1: [0, 0, 3, 4]},
    )

    metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
        tracker, CHUNK_TOKENS, HYBRID_GROUP_TOKENS
    )

    assert metadata is not None
    assert metadata.op.block_ids == [[13, 13, 13], [-1, -1, 3]]
