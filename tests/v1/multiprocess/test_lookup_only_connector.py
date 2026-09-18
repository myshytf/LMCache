# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side connector: lookup-only for local hits on hybrid engines.

With recurrent (Mamba/KDA) cache groups the connector never splices an
external tail onto a local prefix, so a request that already has a local hit
only needs the lookup's hit length to anchor its store bookkeeping. Such a
request must ask for a lookup-only report, must not retrieve, and must not
free locks it never held. A request without a local hit keeps the loading
lookup so cold restores are unchanged.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (
    LMCacheMPConnector,
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)
from vllm.v1.outputs import KVConnectorOutput

CHUNK = 4608


def _tracker(num_tokens: int) -> LMCacheMPRequestTracker:
    tracker = LMCacheMPRequestTracker.__new__(LMCacheMPRequestTracker)
    tracker.request_id = "req-1"
    tracker.cache_salt = ""
    tracker.all_token_ids = list(range(num_tokens))
    tracker.allocated_block_ids = {}
    tracker.num_stored_tokens = 0
    tracker.num_vllm_hit_tokens = 0
    tracker.num_lmcache_hit_tokens = 0
    tracker.skip_mixed_recurrent_retrieve = False
    tracker.lookup_only = False
    tracker.lookup_submitted = False
    tracker.lookup_reissued = False
    tracker.num_admitted_external_tokens = -1
    tracker.state = LMCacheMPRequestState.PREFETCHING
    return tracker


def _connector(tracker: LMCacheMPRequestTracker, lookup_result: int):
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector._has_recurrent_cache = True
    connector._hit_alignment_tokens = CHUNK
    connector.scheduler_adapter = MagicMock()
    connector.scheduler_adapter.lmcache_tokens_per_chunk = CHUNK
    connector.scheduler_adapter.check_lookup_result.return_value = lookup_result
    connector.request_trackers = {tracker.request_id: tracker}
    connector._get_or_create_request_tracker = lambda request: tracker
    return connector


def _request():
    return SimpleNamespace(request_id="req-1", status="WAITING")


@pytest.mark.parametrize("num_computed_tokens", [CHUNK, 3 * CHUNK])
def test_local_hit_requests_a_lookup_only_report(num_computed_tokens):
    tracker = _tracker(num_tokens=5 * CHUNK + 100)
    connector = _connector(tracker, lookup_result=4 * CHUNK)

    result = connector.get_num_new_matched_tokens(_request(), num_computed_tokens)

    submit = connector.scheduler_adapter.maybe_submit_lookup_request
    submit.assert_called_once()
    assert submit.call_args.kwargs["lookup_only"] is True
    assert tracker.lookup_only is True
    # The hit still anchors store bookkeeping, but nothing is retrieved.
    assert tracker.num_stored_tokens == 4 * CHUNK
    assert tracker.num_lmcache_hit_tokens == 4 * CHUNK
    assert tracker.skip_mixed_recurrent_retrieve is (4 * CHUNK > num_computed_tokens)
    assert result == (0, False)


def test_cold_request_keeps_the_loading_lookup():
    tracker = _tracker(num_tokens=5 * CHUNK + 100)
    connector = _connector(tracker, lookup_result=4 * CHUNK)

    result = connector.get_num_new_matched_tokens(_request(), 0)

    submit = connector.scheduler_adapter.maybe_submit_lookup_request
    assert submit.call_args.kwargs["lookup_only"] is False
    assert tracker.lookup_only is False
    assert result == (4 * CHUNK, True)


def test_lookup_only_tracker_frees_no_locks_after_allocation():
    tracker = _tracker(num_tokens=5 * CHUNK + 100)
    tracker.lookup_only = True
    tracker.num_vllm_hit_tokens = CHUNK
    tracker.num_lmcache_hit_tokens = 4 * CHUNK
    tracker.skip_mixed_recurrent_retrieve = True
    connector = _connector(tracker, lookup_result=4 * CHUNK)
    connector._get_request_tracker = lambda request_id: tracker
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([1, 2],)

    connector.update_state_after_alloc(_request(), blocks, 0)

    assert tracker.state == LMCacheMPRequestState.READY
    connector.scheduler_adapter.free_lookup_locks.assert_not_called()
    connector.scheduler_adapter.cleanup_lookup_result.assert_called_once_with("req-1")


def test_loading_tracker_still_frees_locks_it_holds():
    tracker = _tracker(num_tokens=5 * CHUNK + 100)
    tracker.num_vllm_hit_tokens = CHUNK
    tracker.num_lmcache_hit_tokens = 4 * CHUNK
    tracker.skip_mixed_recurrent_retrieve = True
    connector = _connector(tracker, lookup_result=4 * CHUNK)
    connector._get_request_tracker = lambda request_id: tracker
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([1, 2],)

    connector.update_state_after_alloc(_request(), blocks, 0)

    assert tracker.state == LMCacheMPRequestState.READY
    free = connector.scheduler_adapter.free_lookup_locks
    free.assert_called_once()
    assert free.call_args.kwargs["end"] == 4 * CHUNK


@pytest.mark.parametrize(
    "state", [LMCacheMPRequestState.WAITING_FOR_LOAD, LMCacheMPRequestState.READY]
)
def test_admitted_request_cannot_enter_a_second_async_load(state) -> None:
    """A recomputed request must not wait for a retrieve that is never emitted."""
    tracker = _tracker(num_tokens=5 * CHUNK + 100)
    tracker.state = state
    connector = _connector(tracker, lookup_result=4 * CHUNK)

    assert connector.get_num_new_matched_tokens(_request(), 0) == (0, False)
    connector.scheduler_adapter.maybe_submit_lookup_request.assert_not_called()


def test_failed_restore_recomputes_and_stores_only_current_blocks() -> None:
    """Freed restore pages and uncomputed hit tokens cannot enter a new store."""
    tracker = _tracker(num_tokens=5 * CHUNK + 100)
    tracker.state = LMCacheMPRequestState.READY
    tracker.allocated_block_ids = {0: [10, 11, 12, 13]}
    tracker.num_lmcache_hit_tokens = tracker.num_stored_tokens = 4 * CHUNK
    tracker.num_admitted_external_tokens = 4 * CHUNK
    connector = _connector(tracker, lookup_result=4 * CHUNK)

    connector.update_connector_output(KVConnectorOutput(invalid_block_ids={12}))
    # All workers still own their pages until the scheduler receives completion.
    assert tracker.allocated_block_ids == {0: [10, 11, 12, 13]}
    assert connector.get_num_new_matched_tokens(_request(), 0) == (0, False)
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([20],)
    connector.update_state_after_alloc(_request(), blocks, 0)
    tracker.anchor_num_scheduled_tokens(0, CHUNK)

    store = LMCacheMPRequestMetadata.GetStoreMetadata(tracker, CHUNK, [CHUNK])
    assert store is not None
    assert (store.op.start, store.op.end) == (0, CHUNK)
    assert store.op.block_ids == [[20]]
    assert tracker.num_lmcache_hit_tokens == 0


def test_unrelated_failure_preserves_successful_restore_bookkeeping() -> None:
    tracker = _tracker(num_tokens=5 * CHUNK + 100)
    tracker.state = LMCacheMPRequestState.READY
    tracker.allocated_block_ids = {0: [10, 11]}
    tracker.num_lmcache_hit_tokens = tracker.num_stored_tokens = 2 * CHUNK
    connector = _connector(tracker, lookup_result=2 * CHUNK)

    connector.update_connector_output(KVConnectorOutput(invalid_block_ids={99}))

    assert tracker.num_lmcache_hit_tokens == tracker.num_stored_tokens == 2 * CHUNK


def test_lookup_kind_is_fixed_while_the_result_is_pending():
    """The first call decides the kind; a later re-query with a different
    local hit neither resubmits nor flips the tracker."""
    tracker = _tracker(num_tokens=5 * CHUNK + 100)
    connector = _connector(tracker, lookup_result=None)

    assert connector.get_num_new_matched_tokens(_request(), CHUNK) == (None, True)
    assert tracker.lookup_only is True
    assert tracker.lookup_submitted is True

    assert connector.get_num_new_matched_tokens(_request(), 0) == (None, True)

    submit = connector.scheduler_adapter.maybe_submit_lookup_request
    assert [c.kwargs["lookup_only"] for c in submit.call_args_list] == [True, True]
    assert tracker.lookup_only is True
    connector.scheduler_adapter.cleanup_lookup_result.assert_not_called()


def test_presence_only_job_is_reissued_as_loading_when_the_local_hit_vanishes():
    """A request whose local prefix was evicted while its presence-only
    lookup was pending must not admit that report as loadable: the report is
    consumed and a loading lookup is issued in its place."""
    tracker = _tracker(num_tokens=50 * CHUNK + 100)
    connector = _connector(tracker, lookup_result=None)
    assert connector.get_num_new_matched_tokens(_request(), CHUNK) == (None, True)

    adapter = connector.scheduler_adapter
    # Presence report arrives; the loading lookup is still pending.
    adapter.check_lookup_result.side_effect = [46 * CHUNK, None]
    assert connector.get_num_new_matched_tokens(_request(), 0) == (None, True)

    adapter.cleanup_lookup_result.assert_called_once_with("req-1")
    # Submission is idempotent while a lookup is pending: the re-query repeats
    # the presence-only kind, then the loading lookup replaces it.
    kinds = [c.kwargs["lookup_only"] for c in adapter.maybe_submit_lookup_request.call_args_list]
    assert kinds == [True, True, False]
    assert tracker.lookup_only is False
    assert tracker.lookup_reissued is True
    # Nothing anchored on the consumed presence report.
    assert tracker.num_stored_tokens == 0

    # The loading lookup resolves: the prefix is admitted for restore.
    adapter.check_lookup_result.side_effect = None
    adapter.check_lookup_result.return_value = 46 * CHUNK
    assert connector.get_num_new_matched_tokens(_request(), 0) == (46 * CHUNK, True)
    assert tracker.num_stored_tokens == 46 * CHUNK
    assert tracker.num_lmcache_hit_tokens == 46 * CHUNK
    assert tracker.skip_mixed_recurrent_retrieve is False
    # Only one re-issue ever happens; the final re-query repeats the loading kind.
    kinds = [c.kwargs["lookup_only"] for c in adapter.maybe_submit_lookup_request.call_args_list]
    assert kinds == [True, True, False, False]
    assert adapter.cleanup_lookup_result.call_count == 1


def test_presence_only_report_with_a_local_hit_is_not_reissued():
    tracker = _tracker(num_tokens=5 * CHUNK + 100)
    connector = _connector(tracker, lookup_result=4 * CHUNK)

    assert connector.get_num_new_matched_tokens(_request(), CHUNK) == (0, False)

    assert tracker.lookup_only is True
    assert tracker.lookup_reissued is False
    connector.scheduler_adapter.cleanup_lookup_result.assert_not_called()
    assert tracker.num_stored_tokens == 4 * CHUNK


def test_reissued_loading_lookup_with_a_returned_local_hit_recomputes_the_tail():
    """If the local prefix reappears after the re-issue, the loading lookup's
    result follows the normal mixed-recurrent rule and its locks are freed
    later by update_state_after_alloc (the tracker is no longer lookup-only)."""
    tracker = _tracker(num_tokens=50 * CHUNK + 100)
    connector = _connector(tracker, lookup_result=None)
    assert connector.get_num_new_matched_tokens(_request(), CHUNK) == (None, True)
    adapter = connector.scheduler_adapter
    adapter.check_lookup_result.side_effect = [46 * CHUNK, None]
    assert connector.get_num_new_matched_tokens(_request(), 0) == (None, True)

    adapter.check_lookup_result.side_effect = None
    adapter.check_lookup_result.return_value = 46 * CHUNK
    assert connector.get_num_new_matched_tokens(_request(), CHUNK) == (0, False)
    assert tracker.lookup_only is False
    assert tracker.skip_mixed_recurrent_retrieve is True


def _blocks(block_ids: list[int]):
    blocks = MagicMock()
    blocks.get_block_ids.return_value = (block_ids,)
    return blocks


def test_stale_mixed_verdict_is_cleared_when_the_presence_report_is_reissued():
    """Production 2026-09-18: a presence report longer than the local hit was
    refused (mixed-recurrent rule) while the request waited for token budget;
    the local prefix was then evicted to zero and the report was re-issued as
    a loading lookup. The stale refusal must not survive the re-issue, or the
    admitted load is never retrieved and the scheduler waits forever."""
    tracker = _tracker(num_tokens=6 * CHUNK + 1464)
    connector = _connector(tracker, lookup_result=6 * CHUNK)
    connector._get_request_tracker = lambda request_id: tracker
    connector._group_tokens_per_block = [CHUNK]

    # Partial local hit: the presence report is refused, nothing is loaded.
    assert connector.get_num_new_matched_tokens(_request(), 4 * CHUNK) == (0, False)
    assert tracker.skip_mixed_recurrent_retrieve is True

    # The local prefix is gone: the report is consumed and re-issued as a
    # loading lookup, which resolves on the next query.
    adapter = connector.scheduler_adapter
    adapter.check_lookup_result.side_effect = [6 * CHUNK, None]
    assert connector.get_num_new_matched_tokens(_request(), 0) == (None, True)
    assert tracker.lookup_reissued is True
    adapter.check_lookup_result.side_effect = None
    adapter.check_lookup_result.return_value = 6 * CHUNK
    assert connector.get_num_new_matched_tokens(_request(), 0) == (6 * CHUNK, True)
    assert tracker.skip_mixed_recurrent_retrieve is False

    # The scheduler admits the whole prefix; the tracker must retrieve it.
    connector.update_state_after_alloc(
        _request(), _blocks([10, 11, 12, 13, 14, 15, 16]), 6 * CHUNK
    )
    assert tracker.state == LMCacheMPRequestState.WAITING_FOR_LOAD
    assert tracker.needs_retrieve() is True
    retrieve = LMCacheMPRequestMetadata.GetRetrieveMetadata(tracker, CHUNK, [CHUNK])
    assert retrieve is not None
    assert (retrieve.op.start, retrieve.op.end) == (0, 6 * CHUNK)
    assert retrieve.op.block_ids == [[10, 11, 12, 13, 14, 15]]
    # Nothing was freed ahead of the retrieve: the whole range is loaded.
    adapter.free_lookup_locks.assert_not_called()


def test_admitted_async_load_without_a_retrieve_is_reported_as_a_failed_load():
    """Whatever refuses the retrieve after the scheduler admitted an async
    load, the request must not be left in WAITING_FOR_REMOTE_KVS: the worker
    is handed the allocation as a suppressed retrieve and reports the load as
    failed, so vLLM recomputes."""
    tracker = _tracker(num_tokens=5 * CHUNK + 100)
    tracker.num_lmcache_hit_tokens = tracker.num_stored_tokens = 4 * CHUNK
    tracker.skip_mixed_recurrent_retrieve = True
    connector = _connector(tracker, lookup_result=4 * CHUNK)
    connector._get_request_tracker = lambda request_id: tracker
    connector._group_tokens_per_block = [CHUNK]

    connector.update_state_after_alloc(_request(), _blocks([10, 11, 12, 13]), 4 * CHUNK)

    assert tracker.state == LMCacheMPRequestState.WAITING_FOR_LOAD
    # Locks of the whole hit are released: none of it will be retrieved.
    free = connector.scheduler_adapter.free_lookup_locks
    free.assert_called_once()
    assert free.call_args.kwargs["end"] == 4 * CHUNK

    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnectorMetadata

    metadata = LMCacheMPConnectorMetadata()
    connector._process_retrieve_requests(metadata)
    assert metadata.requests == []
    assert metadata.suppressed_retrieves == [("req-1", [10, 11, 12, 13])]
    assert tracker.state == LMCacheMPRequestState.READY


def test_admitting_nothing_still_goes_ready_without_a_retrieve():
    """The safety net only fires for admitted loads; a zero admission keeps the
    existing behaviour."""
    tracker = _tracker(num_tokens=5 * CHUNK + 100)
    tracker.num_lmcache_hit_tokens = 4 * CHUNK
    tracker.skip_mixed_recurrent_retrieve = True
    connector = _connector(tracker, lookup_result=4 * CHUNK)
    connector._get_request_tracker = lambda request_id: tracker

    connector.update_state_after_alloc(_request(), _blocks([10, 11]), 0)

    assert tracker.state == LMCacheMPRequestState.READY
