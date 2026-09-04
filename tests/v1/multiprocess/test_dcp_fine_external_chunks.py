# SPDX-License-Identifier: Apache-2.0
"""Tests for DCP manager blocks projected onto fine LMCache objects."""

# Standard
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess import custom_types
from lmcache.v1.multiprocess.group_view import (
    SKIPPED_BLOCK_ID,
    EngineGroupInfo,
    chunk_block_ids,
    first_chunk_using_block,
    mark_block_id_skipped,
    slice_block_ids_per_group,
)
from lmcache.v1.multiprocess.protocols.engine import (
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenContextPickle,
    EngineDrivenTransferContext,
    make_external_subblock_view,
    project_external_chunk_block_ids,
    project_external_chunk_skip_tokens,
    resolve_external_chunk_group_geometry,
)


def test_slice_manager_ids_at_external_chunk_boundaries() -> None:
    allocated = {
        0: list(range(10, 14)),
        1: list(range(28)),
    }

    sliced = slice_block_ids_per_group(
        allocated,
        [12_288, 1_536],
        start_token_idx=36_864,
        end_token_idx=43_008,
        external_chunk_size=1_536,
    )

    assert sliced == [[13, 13, 13, 13], [24, 25, 26, 27]]


def test_slice_fine_range_rejects_missing_manager_block() -> None:
    with pytest.raises(ValueError, match="does not contain manager block index 3"):
        slice_block_ids_per_group(
            {0: [10, 11, 12]},
            [12_288],
            start_token_idx=36_864,
            end_token_idx=38_400,
            external_chunk_size=1_536,
        )


@pytest.mark.parametrize(
    ("start_token_idx", "end_token_idx"), [(-1_536, 0), (1_536, 0)]
)
def test_slice_rejects_invalid_token_range(
    start_token_idx: int, end_token_idx: int
) -> None:
    with pytest.raises(ValueError, match="invalid token range"):
        slice_block_ids_per_group(
            {0: [10]},
            [12_288],
            start_token_idx=start_token_idx,
            end_token_idx=end_token_idx,
            external_chunk_size=1_536,
        )


def test_slice_rejects_range_misaligned_to_external_chunk() -> None:
    with pytest.raises(ValueError, match="external chunk size 3072"):
        slice_block_ids_per_group(
            {0: [10, 11, 12]},
            [1_536],
            start_token_idx=1_536,
            end_token_idx=4_608,
            external_chunk_size=3_072,
        )


@pytest.mark.parametrize("external_chunk_size", [0, -1_536, 10_000])
def test_slice_rejects_invalid_external_geometry(external_chunk_size: int) -> None:
    with pytest.raises(ValueError, match="external chunk size"):
        slice_block_ids_per_group(
            {0: list(range(20))},
            [1_536],
            start_token_idx=0,
            end_token_idx=30_720,
            external_chunk_size=external_chunk_size,
        )


def test_slice_rejects_nonpositive_group_block_span() -> None:
    with pytest.raises(ValueError, match="tokens_per_block must be positive"):
        slice_block_ids_per_group(
            {0: [1]},
            [0],
            start_token_idx=0,
            end_token_idx=0,
        )


def test_project_manager_ids_to_rank_local_subblocks() -> None:
    projected, rows = project_external_chunk_block_ids(
        [13, 13, 13, 13],
        start_token_idx=36_864,
        external_chunk_size=1_536,
        logical_tokens_per_block=12_288,
        physical_tokens_per_block=1_536,
    )

    assert projected == [104, 105, 106, 107]
    assert rows == 192


def test_project_rejects_manager_change_before_subblock_wrap() -> None:
    with pytest.raises(ValueError, match="changed before external sub-block wrap"):
        project_external_chunk_block_ids(
            [13, 14, 14, 14],
            start_token_idx=36_864,
            external_chunk_size=1_536,
            logical_tokens_per_block=12_288,
            physical_tokens_per_block=1_536,
        )


def test_project_requires_manager_change_at_subblock_wrap() -> None:
    with pytest.raises(ValueError, match="did not change at external sub-block wrap"):
        project_external_chunk_block_ids(
            [13] * 9,
            start_token_idx=36_864,
            external_chunk_size=1_536,
            logical_tokens_per_block=12_288,
            physical_tokens_per_block=1_536,
        )


def test_project_skip_tokens_to_rank_local_rows() -> None:
    assert (
        project_external_chunk_skip_tokens(
            1_536,
            logical_tokens_per_block=12_288,
            physical_tokens_per_block=1_536,
            subblocks_per_manager=8,
        )
        == 192
    )


def test_project_rejects_skip_inside_external_chunk() -> None:
    with pytest.raises(ValueError, match="must align to external chunk span 1536"):
        project_external_chunk_skip_tokens(
            768,
            logical_tokens_per_block=12_288,
            physical_tokens_per_block=1_536,
            subblocks_per_manager=8,
        )


def test_project_coarse_dcp_skip_requires_manager_boundary() -> None:
    with pytest.raises(ValueError, match="must align to external chunk span 12288"):
        project_external_chunk_skip_tokens(
            1_536,
            logical_tokens_per_block=12_288,
            physical_tokens_per_block=1_536,
            subblocks_per_manager=1,
        )

    assert (
        project_external_chunk_skip_tokens(
            12_288,
            logical_tokens_per_block=12_288,
            physical_tokens_per_block=1_536,
            subblocks_per_manager=1,
        )
        == 1_536
    )


def test_external_subblock_view_is_zero_copy() -> None:
    source = torch.arange(2 * 8 * 3).reshape(2, 8, 3)

    view = make_external_subblock_view(
        {"layer": source},
        subblocks_per_manager=4,
        physical_tokens_per_block=8,
    )["layer"]

    assert view.shape == (8, 2, 3)
    assert view.data_ptr() == source.data_ptr()
    torch.testing.assert_close(view[6], source[1, 4:6])


def test_fine_geometry_rejects_sliding_window_state() -> None:
    with pytest.raises(ValueError, match="cannot split sliding-window or recurrent"):
        resolve_external_chunk_group_geometry(
            external_chunk_size=1_536,
            logical_tokens_per_block=12_288,
            physical_tokens_per_block=1_536,
            sliding_window_tokens=1_536,
        )


class _CapturingGroupedContext:
    def __init__(self) -> None:
        self.prepared = False
        self.committed = False

    def prepare_store_grouped(self, _key: Any, _instance_id: int):
        self.prepared = True
        return (
            [torch.zeros(1) for _ in range(8)],
            [0, 1, 2, 3, 0, 1, 2, 3],
            [0, 0, 0, 0, 1, 1, 1, 1],
        )

    def commit_store(self, _key: Any, _instance_id: int, _chunks: Any) -> bool:
        self.committed = True
        return True

    def prepare_retrieve_grouped(self, _key: Any, _instance_id: int):
        return (
            [torch.zeros(1) for _ in range(8)],
            [0, 0, 0, 0, 1, 1, 1, 1],
        )

    def commit_retrieve(self, _key: Any, _instance_id: int) -> bool:
        return True

    def close(self) -> None:
        return None


def _register_fine_context(
    monkeypatch: pytest.MonkeyPatch,
    transfer_context: _CapturingGroupedContext,
) -> tuple[EngineDrivenTransferContext, dict[str, torch.Tensor]]:
    registration = MagicMock()
    registration.result.return_value = RegisterEngineDrivenContextResponse()
    monkeypatch.setattr(
        worker_transfer,
        "create_engine_driven_context",
        lambda *_args, **_kwargs: transfer_context,
    )
    monkeypatch.setattr(worker_transfer.torch_dev, "synchronize", lambda: None)
    context = EngineDrivenTransferContext()
    kv_caches = {
        "attention": torch.zeros(30, 8, 1),
        "recurrent": torch.zeros(30, 8, 1),
    }
    context.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=1,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=MagicMock(return_value=registration),
        engine_group_infos=[
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(0,),
                tokens_per_block=64,
                sw_size_tokens=-1,
            ),
            EngineGroupInfo(
                engine_group_id=1,
                layer_indices=(1,),
                tokens_per_block=8,
                sw_size_tokens=8,
            ),
        ],
    )
    return context, kv_caches


def _fine_key() -> custom_types.IPCCacheServerKey:
    return custom_types.IPCCacheServerKey.from_token_ids(
        "m",
        1,
        0,
        [1] * 224,
        start=192,
        end=224,
        request_id="request",
    )


def test_submit_store_projects_only_fine_attention_group(monkeypatch) -> None:
    capturing_context = _CapturingGroupedContext()
    context, kv_caches = _register_fine_context(monkeypatch, capturing_context)
    calls: list[dict[str, Any]] = []

    def record_gather(
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        blocks_in_chunk: int,
        **kwargs: Any,
    ) -> None:
        tensor = next(iter(kv_caches.values()))
        calls.append(
            {
                "shape": tuple(tensor.shape),
                "block_ids": block_ids,
                "blocks_in_chunk": blocks_in_chunk,
                "chunk_indices": kwargs["chunk_indices"],
            }
        )

    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", record_gather)
    key = _fine_key()

    future = context.submit_store(
        "request",
        key,
        1,
        kv_caches,
        [[13, 13, 13, 13], [24, 25, 26, 27]],
        MagicMock(),
        1,
    )

    assert future.result(timeout=1) is True
    assert capturing_context.committed is True
    assert calls == [
        {
            "shape": (240, 1, 1),
            "block_ids": [104, 105, 106, 107],
            "blocks_in_chunk": 1,
            "chunk_indices": [0, 1, 2, 3],
        },
        {
            "shape": (30, 8, 1),
            "block_ids": [24, 25, 26, 27],
            "blocks_in_chunk": 1,
            "chunk_indices": [0, 1, 2, 3],
        },
    ]


def test_submit_store_rejects_block_ids_shorter_than_key_span(monkeypatch) -> None:
    capturing_context = _CapturingGroupedContext()
    context, kv_caches = _register_fine_context(monkeypatch, capturing_context)

    with pytest.raises(ValueError, match="requires 4 block IDs, got 1"):
        context.submit_store(
            "request",
            _fine_key(),
            1,
            kv_caches,
            [[13], [24, 25, 26, 27]],
            MagicMock(),
            1,
        )

    assert capturing_context.prepared is False


def test_submit_retrieve_projects_fine_ids_and_skip_rows(monkeypatch) -> None:
    context, kv_caches = _register_fine_context(monkeypatch, _CapturingGroupedContext())
    calls: list[dict[str, Any]] = []

    def record_scatter(
        group_kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        _chunks: list[torch.Tensor],
        blocks_in_chunk: int,
        **kwargs: Any,
    ) -> None:
        tensor = next(iter(group_kv_caches.values()))
        calls.append(
            {
                "shape": tuple(tensor.shape),
                "block_ids": block_ids,
                "blocks_in_chunk": blocks_in_chunk,
                "skip_first_n_tokens": kwargs["skip_first_n_tokens"],
            }
        )

    monkeypatch.setattr(worker_transfer, "scatter_cpu_to_paged_kv", record_scatter)
    future = context.submit_retrieve(
        "request",
        _fine_key(),
        1,
        kv_caches,
        [[13, 13, 13, 13], [24, 25, 26, 27]],
        MagicMock(),
        1,
        skip_first_n_tokens=8,
    )

    assert future.result(timeout=1) is True
    assert calls == [
        {
            "shape": (240, 1, 1),
            "block_ids": [104, 105, 106, 107],
            "blocks_in_chunk": 1,
            "skip_first_n_tokens": 1,
        },
        {
            "shape": (30, 8, 1),
            "block_ids": [24, 25, 26, 27],
            "blocks_in_chunk": 1,
            "skip_first_n_tokens": 8,
        },
    ]


def test_pickle_store_uses_fine_group_projection(monkeypatch) -> None:
    context, kv_caches = _register_fine_context(monkeypatch, _CapturingGroupedContext())
    pickle_context = EngineDrivenContextPickle(
        metadata=MagicMock(), mq_client=MagicMock(), mq_timeout=0.1
    )
    context._engine_driven_context = pickle_context
    calls: list[dict[str, Any]] = []

    def record_gather(
        group_kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        blocks_in_chunk: int,
        **_kwargs: Any,
    ) -> list[torch.Tensor]:
        tensor = next(iter(group_kv_caches.values()))
        calls.append(
            {
                "shape": tuple(tensor.shape),
                "block_ids": block_ids,
                "blocks_in_chunk": blocks_in_chunk,
            }
        )
        return [torch.zeros(1) for _ in block_ids]

    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", record_gather)
    monkeypatch.setattr(pickle_context, "prepare_store", lambda _key, _instance: None)
    monkeypatch.setattr(
        pickle_context,
        "commit_store",
        lambda _key, _instance, _chunks: True,
    )

    future = context.submit_store(
        "request",
        _fine_key(),
        1,
        kv_caches,
        [[13, 13, 13, 13], [24, 25, 26, 27]],
        MagicMock(),
        1,
    )

    assert future.result(timeout=1) is True
    assert calls == [
        {
            "shape": (240, 1, 1),
            "block_ids": [104, 105, 106, 107],
            "blocks_in_chunk": 1,
        },
        {
            "shape": (30, 8, 1),
            "block_ids": [24, 25, 26, 27],
            "blocks_in_chunk": 1,
        },
    ]


def test_pickle_retrieve_uses_fine_group_projection(monkeypatch) -> None:
    context, kv_caches = _register_fine_context(monkeypatch, _CapturingGroupedContext())
    pickle_context = EngineDrivenContextPickle(
        metadata=MagicMock(), mq_client=MagicMock(), mq_timeout=0.1
    )
    context._engine_driven_context = pickle_context
    payload = [
        [torch.zeros(1) for _ in range(4)],
        [torch.zeros(1) for _ in range(4)],
    ]
    monkeypatch.setattr(
        pickle_context,
        "prepare_retrieve_multigroup",
        lambda _key, _instance: payload,
    )
    calls: list[dict[str, Any]] = []

    def record_scatter(
        group_kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        _chunks: list[torch.Tensor],
        blocks_in_chunk: int,
        **kwargs: Any,
    ) -> None:
        tensor = next(iter(group_kv_caches.values()))
        calls.append(
            {
                "shape": tuple(tensor.shape),
                "block_ids": block_ids,
                "blocks_in_chunk": blocks_in_chunk,
                "skip_first_n_tokens": kwargs["skip_first_n_tokens"],
            }
        )

    monkeypatch.setattr(worker_transfer, "scatter_cpu_to_paged_kv", record_scatter)
    future = context.submit_retrieve(
        "request",
        _fine_key(),
        1,
        kv_caches,
        [[13, 13, 13, 13], [24, 25, 26, 27]],
        MagicMock(),
        1,
        skip_first_n_tokens=8,
    )

    assert future.result(timeout=1) is True
    assert calls == [
        {
            "shape": (240, 1, 1),
            "block_ids": [104, 105, 106, 107],
            "blocks_in_chunk": 1,
            "skip_first_n_tokens": 1,
        },
        {
            "shape": (30, 8, 1),
            "block_ids": [24, 25, 26, 27],
            "blocks_in_chunk": 1,
            "skip_first_n_tokens": 8,
        },
    ]


def test_submit_retrieve_drops_skipped_recurrent_prefix(monkeypatch) -> None:
    """Chunks whose recurrent destination is marked skipped (a null placeholder
    block) are not scattered for that group, while the attention group still
    loads the whole range."""
    context, kv_caches = _register_fine_context(monkeypatch, _CapturingGroupedContext())
    calls: list[dict[str, Any]] = []

    def record_scatter(
        group_kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        chunks: list[torch.Tensor],
        blocks_in_chunk: int,
        **kwargs: Any,
    ) -> None:
        tensor = next(iter(group_kv_caches.values()))
        calls.append(
            {
                "shape": tuple(tensor.shape),
                "block_ids": block_ids,
                "num_chunks": len(chunks),
                "skip_first_n_tokens": kwargs["skip_first_n_tokens"],
            }
        )

    monkeypatch.setattr(worker_transfer, "scatter_cpu_to_paged_kv", record_scatter)
    future = context.submit_retrieve(
        "request",
        _fine_key(),
        1,
        kv_caches,
        [[13, 13, 13, 13], [-1, -1, 26, 27]],
        MagicMock(),
        1,
        skip_first_n_tokens=8,
    )

    assert future.result(timeout=1) is True
    assert calls == [
        {
            "shape": (240, 1, 1),
            "block_ids": [104, 105, 106, 107],
            "num_chunks": 4,
            "skip_first_n_tokens": 1,
        },
        {
            "shape": (30, 8, 1),
            "block_ids": [26, 27],
            "num_chunks": 2,
            "skip_first_n_tokens": 0,
        },
    ]


def test_submit_retrieve_skips_group_whose_range_is_fully_skipped(monkeypatch) -> None:
    context, kv_caches = _register_fine_context(monkeypatch, _CapturingGroupedContext())
    shapes: list[tuple[int, ...]] = []

    def record_scatter(group_kv_caches: dict[str, torch.Tensor], *_args, **_kwargs) -> None:
        shapes.append(tuple(next(iter(group_kv_caches.values())).shape))

    monkeypatch.setattr(worker_transfer, "scatter_cpu_to_paged_kv", record_scatter)
    future = context.submit_retrieve(
        "request",
        _fine_key(),
        1,
        kv_caches,
        [[13, 13, 13, 13], [-1, -1, -1, -1]],
        MagicMock(),
        1,
    )

    assert future.result(timeout=1) is True
    assert shapes == [(240, 1, 1)]


def test_submit_retrieve_fails_closed_on_interior_skipped_chunk(monkeypatch) -> None:
    """A skipped destination after a kept chunk cannot be expressed as a
    contiguous suffix; the retrieve reports failure so the engine recomputes."""
    context, kv_caches = _register_fine_context(monkeypatch, _CapturingGroupedContext())
    monkeypatch.setattr(
        worker_transfer, "scatter_cpu_to_paged_kv", lambda *_a, **_k: None
    )
    future = context.submit_retrieve(
        "request",
        _fine_key(),
        1,
        kv_caches,
        [[13, 13, 13, 13], [24, -1, 26, 27]],
        MagicMock(),
        1,
    )

    assert future.result(timeout=1) is False


@pytest.mark.parametrize(
    ("block_ids", "blocks_in_chunk", "expected_ids", "expected_dropped"),
    [
        ([-1, -1, 26, 27], 1, [26, 27], 2),
        ([24, 25, 26, 27], 1, [24, 25, 26, 27], 0),
        ([-1, -1, -1, -1], 1, [], 4),
        ([-1, 5, 6, 7, 8, 9, 10, 11], 2, [6, 7, 8, 9, 10, 11], 1),
    ],
)
def test_drop_skipped_chunks_removes_leading_prefix(
    block_ids: list[int],
    blocks_in_chunk: int,
    expected_ids: list[int],
    expected_dropped: int,
) -> None:
    chunks = [torch.full((1,), float(i)) for i in range(len(block_ids) // blocks_in_chunk)]

    kept, ids, dropped = worker_transfer._drop_skipped_chunks(
        chunks, block_ids, blocks_in_chunk
    )

    assert ids == expected_ids
    assert dropped == expected_dropped
    assert [float(c[0]) for c in kept] == [float(i) for i in range(dropped, len(chunks))]


def test_drop_skipped_chunks_rejects_interior_skip() -> None:
    chunks = [torch.zeros(1) for _ in range(3)]
    with pytest.raises(ValueError, match="only leading chunks"):
        worker_transfer._drop_skipped_chunks(chunks, [24, -1, 26], 1)


def test_chunk_block_ids_groups_coarse_and_fine_ids_per_chunk() -> None:
    assert chunk_block_ids([13, 13, 13], 12_288, 1_536, 3) == [[13], [13], [13]]
    assert chunk_block_ids([4, 5, 6, 7], 768, 1_536, 2) == [[4, 5], [6, 7]]
    with pytest.raises(ValueError, match="do not cover"):
        chunk_block_ids([4, 5, 6], 768, 1_536, 2)


def test_first_chunk_using_block_reports_earliest_group_match() -> None:
    sliced = [[13, 13, 13], [0, 0, 3]]
    assert first_chunk_using_block(sliced, [12_288, 1_536], 1_536, 3, 0) == 0
    assert first_chunk_using_block([[13, 13, 13], [1, 0, 3]], [12_288, 1_536], 1_536, 3, 0) == 1
    assert first_chunk_using_block([[13, 13, 13], [1, 2, 3]], [12_288, 1_536], 1_536, 3, 0) is None


def test_mark_block_id_skipped_substitutes_only_that_id() -> None:
    assert mark_block_id_skipped([[13, 13, 13], [0, 0, 3]], 0) == [
        [13, 13, 13],
        [SKIPPED_BLOCK_ID, SKIPPED_BLOCK_ID, 3],
    ]
