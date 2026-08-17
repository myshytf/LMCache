# SPDX-License-Identifier: Apache-2.0
# Third Party
import msgspec

# First Party
from lmcache.v1.multiprocess import group_view
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    expand_engine_block_ids,
    get_engine_group_indices,
    num_engine_group_infos,
    num_engine_groups,
    slice_block_ids_per_group,
)


def test_engine_group_infos_default_to_one_engine_group():
    assert num_engine_groups([]) == 1
    assert num_engine_group_infos([]) == 1
    assert get_engine_group_indices([], 1) is None


def test_engine_group_infos_build_per_layer_engine_group_indices():
    groups = [
        EngineGroupInfo(0, (0, 2)),
        EngineGroupInfo(1, (1, 3)),
    ]

    assert num_engine_groups(groups) == 2
    assert num_engine_group_infos(groups) == 2
    assert get_engine_group_indices(groups, 4) == [0, 1, 0, 1]


def test_engine_group_infos_expand_engine_block_ids():
    groups = [
        EngineGroupInfo(0, (0, 2)),
        EngineGroupInfo(0, (4,)),
        EngineGroupInfo(1, (1, 3)),
    ]

    assert expand_engine_block_ids(groups, [[10, 11], [20, 21]]) == [
        [10, 11],
        [10, 11],
        [20, 21],
    ]


def test_engine_group_infos_expand_virtual_kernel_blocks():
    groups = [
        EngineGroupInfo(0, (0,), physical_blocks_per_engine_block=1),
        EngineGroupInfo(1, (1,), physical_blocks_per_engine_block=2),
    ]

    assert expand_engine_block_ids(groups, [[5, 6], [10, 11]]) == [
        [5, 6],
        [20, 21, 22, 23],
    ]


def test_engine_group_infos_reject_invalid_physical_multiplier():
    groups = [EngineGroupInfo(0, (0,), physical_blocks_per_engine_block=0)]

    try:
        expand_engine_block_ids(groups, [[5]])
    except ValueError as exc:
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("Expected invalid physical block multiplier to fail")


def test_engine_group_info_old_payload_defaults_sw_size():
    """Older msgspec payloads decode with protocol-compatible defaults."""
    old_payload = {"engine_group_id": 0, "layer_indices": (0, 1)}

    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(old_payload), type=EngineGroupInfo
    )

    assert decoded.sw_size_tokens == -1
    assert decoded.physical_blocks_per_engine_block == 1


def test_engine_group_infos_msgspec_round_trip():
    """The groups encode/decode losslessly via msgspec (the IPC path)."""
    groups = [
        EngineGroupInfo(0, (0, 2)),
        EngineGroupInfo(1, (1, 3), sw_size_tokens=128),
    ]

    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(groups), type=list[EngineGroupInfo]
    )

    assert decoded == groups


def test_engine_group_infos_exclude_uncovered_layers():
    """Layers not referenced by any group are tagged EXCLUDED_ENGINE_GROUP.

    Cross-layer KV-sharing layers (e.g. google/gemma-4-E4B-it) alias a target
    owner's KV cache and are intentionally left out of every group; downstream
    grouping skips them rather than treating partial coverage as an error.
    """
    # First Party
    from lmcache.v1.kv_layer_groups import EXCLUDED_ENGINE_GROUP

    groups = [
        EngineGroupInfo(0, (0,)),
        EngineGroupInfo(1, (1,)),
    ]

    # Layer 2 is not covered by any group -> excluded, not an error.
    assert get_engine_group_indices(groups, 3) == [0, 1, EXCLUDED_ENGINE_GROUP]


def test_engine_group_infos_reject_out_of_range_layer():
    groups = [EngineGroupInfo(0, (0, 5))]

    try:
        get_engine_group_indices(groups, 3)
    except ValueError as exc:
        assert "outside registered layer range" in str(exc)
    else:
        raise AssertionError("Expected out-of-range layer index to fail")


def test_slice_block_ids_uniform_block_sizes():
    """Groups sharing one tokens_per_block slice to equal counts."""
    allocated = {0: list(range(16)), 1: list(range(100, 116))}
    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=[16, 16],
        start_token_idx=0,
        end_token_idx=256,
    )
    assert sliced == [list(range(16)), list(range(100, 116))]


def test_slice_block_ids_heterogeneous_block_sizes():
    """A tokens_per_block-32 group gets half the IDs of a 16 group.

    The range [0, 256) spans 256 tokens: the tokens_per_block-16 group
    needs 16 block IDs, the tokens_per_block-32 group 8, for the same
    token span.
    """
    allocated = {0: list(range(16)), 1: list(range(8))}
    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=[16, 32],
        start_token_idx=0,
        end_token_idx=256,
    )
    assert sliced == [list(range(16)), list(range(8))]


def test_slice_block_ids_smaller_than_base_block_sizes():
    """Groups with tiny paged chunks (e.g. DeepSeek V4 compressor state,
    tokens_per_block 4/8) get proportionally more IDs over one token span."""
    allocated = {0: [0], 1: list(range(64)), 2: list(range(32))}
    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=[256, 4, 8],
        start_token_idx=0,
        end_token_idx=256,
    )
    assert sliced == [[0], list(range(64)), list(range(32))]


def test_slice_block_ids_nonzero_start_offset():
    """Start/end token offsets are divided per group by tokens_per_block."""
    allocated = {0: list(range(32)), 1: list(range(16))}
    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=[16, 32],
        start_token_idx=256,
        end_token_idx=512,
    )
    assert sliced == [list(range(16, 32)), list(range(8, 16))]


def test_slice_block_ids_repeats_manager_id_for_finer_external_chunks():
    allocated = {
        0: [10, 11, 12, 13],
        1: list(range(30)),
    }

    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=[12_288, 1_536],
        start_token_idx=36_864,
        end_token_idx=43_008,
        external_chunk_size=1_536,
    )

    assert sliced == [[13, 13, 13, 13], [24, 25, 26, 27]]


def test_slice_fine_range_rejects_missing_manager_block():
    try:
        slice_block_ids_per_group(
            {0: [10, 11, 12]},
            group_tokens_per_block=[12_288],
            start_token_idx=36_864,
            end_token_idx=38_400,
            external_chunk_size=1_536,
        )
    except ValueError as exc:
        assert "does not contain manager block index 3" in str(exc)
    else:
        raise AssertionError("Expected incomplete manager table to fail")


def test_external_chunk_geometry_accepts_fine_dcp_groups():
    assert group_view.validate_external_chunk_geometry(1_536, [12_288, 1_536]) == 1_536


def test_external_chunk_geometry_rejects_incommensurate_group():
    try:
        group_view.validate_external_chunk_geometry(1_536, [10_000])
    except ValueError as exc:
        assert "must divide each other exactly" in str(exc)
    else:
        raise AssertionError("Expected incommensurate geometry to fail")


def test_slice_block_ids_missing_group_yields_empty():
    """A group with no allocated block IDs slices to an empty list."""
    allocated = {0: list(range(16))}  # group 1 absent
    sliced = slice_block_ids_per_group(
        allocated,
        group_tokens_per_block=[16, 16],
        start_token_idx=0,
        end_token_idx=256,
    )
    assert sliced == [list(range(16)), []]


def test_slice_block_ids_misaligned_range_raises():
    """A range that is not a whole number of a group's chunks is rejected."""
    allocated = {0: list(range(8)), 1: list(range(8))}
    # group 1 tokens_per_block 48; end=128 is not a multiple of 48.
    try:
        slice_block_ids_per_group(
            allocated,
            group_tokens_per_block=[16, 48],
            start_token_idx=0,
            end_token_idx=128,
        )
    except ValueError as exc:
        assert "does not align" in str(exc)
    else:
        raise AssertionError("Expected misaligned range to fail")
