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
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)

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
