# SPDX-License-Identifier: Apache-2.0
"""LMCache's engine-neutral description of a serving engine's KV cache groups.

An *engine group* is one distinct paged-block address space exposed by the
serving engine (e.g. one of vLLM's hybrid KV cache groups): block IDs are only
meaningful within a single group, and layers from different groups must never be
merged into one LMCache KV group. Engine group ids are assumed dense and
consecutive starting from 0.

LMCache's neutral KV cache spec is simply a ``list[EngineGroupInfo]`` (passed as
a ``Sequence[EngineGroupInfo]`` where only order matters). The group order is
the protocol-visible LMCache group order used by store/retrieve block IDs. An
empty list means a single non-hybrid group (the default for engines that do not
report KV cache group metadata). Engine-specific conversion belongs in the
corresponding ``lmcache.integration.<engine>`` package, not here.
"""

# Standard
from collections.abc import Mapping, Sequence
from typing import cast

# Third Party
import msgspec


class EngineGroupInfo(msgspec.Struct, frozen=True):
    """One LMCache KV group: layers of one engine group that share a copy kernel.

    Carries the layer indices and which engine group they belong to. Several
    ``EngineGroupInfo`` instances may share the same ``engine_group_id`` when
    one engine group is split by physical transfer identity (e.g. differing
    hidden dims). A ``list[EngineGroupInfo]`` is carried verbatim in the
    ``REGISTER_KV_CACHE`` IPC payload; the message queue handles
    encoding/decoding.
    """

    engine_group_id: int
    """Engine group these layers live in (one distinct paged-block address
    space). Selects which request block-id list applies. Dense from 0."""

    layer_indices: tuple[int, ...] = ()
    """Registered KV tensor indices assigned to this group."""

    tokens_per_block: int = 0
    """Global sequence tokens covered by one paged chunk (one engine block
    ID) of this group. This normally equals the engine KV spec's block size;
    sequence-sharded context parallel groups multiply it by their shard count,
    while replicated groups do not. ``0`` means the engine did not report it;
    consumers then fall back to the physical slot count detected from the
    registered tensors (i.e. the group is treated as uncompressed)."""

    sw_size_tokens: int = -1
    """Sliding window size in tokens for the layers of this group.
    ``-1`` means the layers are not sliding-window attention."""

    physical_blocks_per_engine_block: int = 1
    """Number of physical kernel blocks backing one engine manager block ID.

    vLLM can split one scheduler-visible block into consecutive physical block
    IDs. ``1`` preserves the historical one-to-one protocol behavior and is
    also the default when decoding payloads from older connectors.
    """


def num_engine_groups(groups: Sequence[EngineGroupInfo]) -> int:
    """Return the number of engine groups (block-id lists per transfer request).

    Engine group ids are assumed dense and consecutive from 0.

    Args:
        groups: The LMCache KV groups, in protocol order.

    Returns:
        ``max(engine_group_id) + 1``, or ``1`` for an empty ``groups`` (single
        non-hybrid group).
    """
    if not groups:
        return 1
    return max(group.engine_group_id for group in groups) + 1


def num_engine_group_infos(groups: Sequence[EngineGroupInfo]) -> int:
    """Return the number of LMCache KV groups visible to transfer requests.

    Args:
        groups: The LMCache KV groups, in protocol order.

    Returns:
        ``len(groups)``, or ``1`` for an empty ``groups`` (single non-hybrid
        group).
    """
    if not groups:
        return 1
    return len(groups)


def engine_group_layer_indices(
    groups: Sequence[EngineGroupInfo],
) -> list[list[int]]:
    """Return each engine group's layer indices, ordered by engine group id.

    Several ``EngineGroupInfo`` may share one ``engine_group_id``; their
    ``layer_indices`` are unioned into that group's entry.

    Args:
        groups: The LMCache KV groups, in protocol order.

    Returns:
        One sorted ``list[int]`` of layer indices per engine group, indexed by
        engine group id (dense from 0). Empty when ``groups`` is empty (a single
        non-hybrid group with no per-group split).
    """
    if not groups:
        return []
    num_groups = max(group.engine_group_id for group in groups) + 1
    per_group: list[list[int]] = [[] for _ in range(num_groups)]
    for group in groups:
        per_group[group.engine_group_id].extend(group.layer_indices)
    return [sorted(indices) for indices in per_group]


def expand_engine_block_ids(
    groups: Sequence[EngineGroupInfo],
    engine_side_block_ids: Sequence[Sequence[int]] | Sequence[int],
) -> list[list[int]]:
    """Expand the engine-side block id list to the list per LMCache kernel group.

    The serving engine reports block IDs per engine group. LMCache transfer
    requests are indexed by LMCache KV group, so each LMCache group reuses the
    block IDs from its source engine group. If the serving engine virtually
    splits one manager block, its ID ``b`` expands to consecutive physical IDs
    ``b * N .. b * N + N - 1`` before LMCache accesses registered tensors.

    Args:
        groups: The LMCache KV groups, in protocol order.
        engine_side_block_ids: Block IDs indexed by engine group id, i.e. one
            inner ``list[int]`` per engine group (element ``g`` is engine group
            ``g``'s block list).

    Returns:
        Block IDs re-indexed by LMCache group order: one inner list per LMCache
        group, expanded to physical kernel block IDs when required.
    """
    # Back-compat: older vLLM connectors emit a flat Sequence[int] for the
    # single (non-hybrid) engine group instead of one inner list per group.
    # Normalize both shapes to a concrete list[list[int]] so downstream
    # indexing is unambiguous for both runtime and mypy.
    if not engine_side_block_ids or isinstance(engine_side_block_ids[0], int):
        per_group: Sequence[Sequence[int]] = [
            cast("Sequence[int]", engine_side_block_ids)
        ]
    else:
        per_group = cast("Sequence[Sequence[int]]", engine_side_block_ids)
    if not groups:
        return [list(per_group[0])]

    expanded: list[list[int]] = []
    for group in groups:
        multiplier = group.physical_blocks_per_engine_block
        if multiplier < 1:
            raise ValueError(
                "physical_blocks_per_engine_block must be positive, got "
                f"{multiplier} for engine group {group.engine_group_id}"
            )
        manager_ids = per_group[group.engine_group_id]
        if multiplier == 1:
            expanded.append(list(manager_ids))
        else:
            expanded.append(
                [
                    block_id * multiplier + offset
                    for block_id in manager_ids
                    for offset in range(multiplier)
                ]
            )
    return expanded


def validate_external_chunk_geometry(
    external_chunk_size: int,
    group_tokens_per_block: Sequence[int],
) -> int:
    """Validate that each group and external object divide one another exactly.

    External objects may contain one or more whole manager blocks, or one
    manager block may span an integer number of finer external objects.

    Args:
        external_chunk_size: Global sequence-token span of one external object.
        group_tokens_per_block: Global sequence-token span represented by one
            manager block ID for each engine group.

    Returns:
        ``external_chunk_size`` after all group geometries validate.

    Raises:
        ValueError: If a size is non-positive or an external object and a
            group's manager block do not divide one another exactly.
    """
    if external_chunk_size <= 0:
        raise ValueError(
            f"external chunk size must be positive, got {external_chunk_size}"
        )
    for engine_group_idx, tokens_per_block in enumerate(group_tokens_per_block):
        if tokens_per_block <= 0:
            raise ValueError(
                f"group {engine_group_idx} tokens_per_block must be positive, "
                f"got {tokens_per_block}"
            )
        if (
            external_chunk_size % tokens_per_block
            and tokens_per_block % external_chunk_size
        ):
            raise ValueError(
                f"LMCache chunk size {external_chunk_size} and group "
                f"{engine_group_idx} tokens_per_block {tokens_per_block} "
                "must divide each other exactly"
            )
    return external_chunk_size


def slice_block_ids_per_group(
    allocated_block_ids: Mapping[int, Sequence[int]],
    group_tokens_per_block: Sequence[int],
    start_token_idx: int,
    end_token_idx: int,
    external_chunk_size: int | None = None,
) -> list[list[int]]:
    """Slice each group's block IDs for a token range.

    When an external object is finer than a manager block, repeat that manager
    ID once per external chunk. The worker later projects each repeated ID to
    its rank-local physical sub-block. Omitting ``external_chunk_size`` keeps
    the historical manager-boundary behavior.

    Args:
        allocated_block_ids: Manager block IDs indexed by engine group ID.
        group_tokens_per_block: Global sequence-token span represented by one
            manager block ID for each engine group.
        start_token_idx: Inclusive global sequence-token offset.
        end_token_idx: Exclusive global sequence-token offset.
        external_chunk_size: Optional global token span of one external object.

    Returns:
        One block-ID list per engine group. Fine external geometry repeats each
        manager ID once for every external object it spans.

    Raises:
        ValueError: If sizes are non-positive, the requested range is
            misaligned, fine geometry is non-integral, or the manager block
            table does not cover the requested range.
    """
    if start_token_idx < 0 or end_token_idx < start_token_idx:
        raise ValueError(f"invalid token range [{start_token_idx}, {end_token_idx})")
    if external_chunk_size is not None and external_chunk_size <= 0:
        raise ValueError(
            f"external chunk size must be positive, got {external_chunk_size}"
        )
    if external_chunk_size is not None and (
        start_token_idx % external_chunk_size or end_token_idx % external_chunk_size
    ):
        raise ValueError(
            f"token range [{start_token_idx}, {end_token_idx}) must align to "
            f"external chunk size {external_chunk_size}"
        )

    sliced: list[list[int]] = []
    for engine_group_idx, tokens_per_block in enumerate(group_tokens_per_block):
        if tokens_per_block <= 0:
            raise ValueError(
                f"group {engine_group_idx} tokens_per_block must be positive, "
                f"got {tokens_per_block}"
            )
        if external_chunk_size is None or external_chunk_size >= tokens_per_block:
            if (
                external_chunk_size is not None
                and external_chunk_size % tokens_per_block
            ):
                raise ValueError(
                    f"external chunk size {external_chunk_size} is not a multiple "
                    f"of group {engine_group_idx} tokens_per_block {tokens_per_block}"
                )
            if start_token_idx % tokens_per_block or end_token_idx % tokens_per_block:
                raise ValueError(
                    f"token range [{start_token_idx}, {end_token_idx}) does not "
                    f"align to group {engine_group_idx} tokens_per_block "
                    f"{tokens_per_block}"
                )
            group_block_ids = allocated_block_ids.get(engine_group_idx, [])
            sliced.append(
                list(
                    group_block_ids[
                        start_token_idx // tokens_per_block : end_token_idx
                        // tokens_per_block
                    ]
                )
            )
            continue

        if tokens_per_block % external_chunk_size:
            raise ValueError(
                f"external chunk size {external_chunk_size} does not divide "
                f"group {engine_group_idx} tokens_per_block {tokens_per_block}"
            )
        if start_token_idx % external_chunk_size or end_token_idx % external_chunk_size:
            raise ValueError(
                f"token range [{start_token_idx}, {end_token_idx}) does not "
                f"align to external chunk size {external_chunk_size}"
            )

        group_block_ids = allocated_block_ids.get(engine_group_idx, [])
        repeated: list[int] = []
        for chunk_start in range(start_token_idx, end_token_idx, external_chunk_size):
            manager_idx = chunk_start // tokens_per_block
            if manager_idx >= len(group_block_ids):
                raise ValueError(
                    f"group {engine_group_idx} block table does not contain "
                    f"manager block index {manager_idx}"
                )
            repeated.append(group_block_ids[manager_idx])
        sliced.append(repeated)
    return sliced


# Manager block id a request-side connector substitutes for a block that has
# no transferable destination or source in one engine group. Worker-side
# transfer code drops the external chunks that map to it for that group only.
SKIPPED_BLOCK_ID = -1


def chunk_block_ids(
    group_block_ids: Sequence[int],
    tokens_per_block: int,
    external_chunk_size: int,
    num_chunks: int,
) -> list[list[int]]:
    """Group one engine group's sliced block ids by external chunk.

    ``group_block_ids`` must come from :func:`slice_block_ids_per_group` for
    the same geometry: coarse groups list ``external_chunk_size //
    tokens_per_block`` ids per chunk, fine groups repeat one manager id per
    chunk.

    Args:
        group_block_ids: Block ids of one engine group for the sliced range.
        tokens_per_block: Global token span of one manager block in the group.
        external_chunk_size: Global token span of one external object.
        num_chunks: Number of external chunks in the sliced range.

    Returns:
        ``num_chunks`` lists, one per external chunk in range order.

    Raises:
        ValueError: If the id count does not match the chunk geometry.
    """
    if tokens_per_block <= 0 or external_chunk_size <= 0 or num_chunks < 0:
        raise ValueError("chunk geometry values must be positive")
    ids_per_chunk = (
        external_chunk_size // tokens_per_block
        if external_chunk_size >= tokens_per_block
        else 1
    )
    if len(group_block_ids) != ids_per_chunk * num_chunks:
        raise ValueError(
            f"{len(group_block_ids)} block ids do not cover {num_chunks} chunks "
            f"of {ids_per_chunk} id(s)"
        )
    return [
        list(group_block_ids[chunk * ids_per_chunk : (chunk + 1) * ids_per_chunk])
        for chunk in range(num_chunks)
    ]


def first_chunk_using_block(
    block_ids_per_group: Sequence[Sequence[int]],
    group_tokens_per_block: Sequence[int],
    external_chunk_size: int,
    num_chunks: int,
    block_id: int,
) -> int | None:
    """Return the first external chunk that maps to ``block_id`` in any group.

    Args:
        block_ids_per_group: Per-group block ids from
            :func:`slice_block_ids_per_group` for the same range.
        group_tokens_per_block: Global token span of one manager block per
            engine group.
        external_chunk_size: Global token span of one external object.
        num_chunks: Number of external chunks in the range.
        block_id: Manager block id to search for.

    Returns:
        The chunk index, or ``None`` when no group maps a chunk to ``block_id``.
    """
    first: int | None = None
    for group_ids, tokens_per_block in zip(
        block_ids_per_group, group_tokens_per_block, strict=True
    ):
        for chunk, ids in enumerate(
            chunk_block_ids(group_ids, tokens_per_block, external_chunk_size, num_chunks)
        ):
            if block_id in ids:
                first = chunk if first is None else min(first, chunk)
                break
    return first


def mark_block_id_skipped(
    block_ids_per_group: Sequence[Sequence[int]],
    block_id: int,
) -> list[list[int]]:
    """Replace every occurrence of ``block_id`` with :data:`SKIPPED_BLOCK_ID`.

    Args:
        block_ids_per_group: Per-group block id lists of one transfer op.
        block_id: Manager block id that must not be transferred.

    Returns:
        New per-group lists with the substitution applied.
    """
    return [
        [SKIPPED_BLOCK_ID if bid == block_id else bid for bid in group_ids]
        for group_ids in block_ids_per_group
    ]


def get_engine_group_indices(
    groups: Sequence[EngineGroupInfo],
    num_registered_layers: int,
) -> list[int] | None:
    """Return the engine group index for each registered KV tensor.

    Args:
        groups: The LMCache KV groups, in protocol order.
        num_registered_layers: Number of KV tensors registered with the server,
            i.e. the length of the per-layer mapping to produce.

    Returns:
        A list of length ``num_registered_layers`` mapping each registered
        tensor index to its engine group id, or ``None`` when there is no group
        metadata (empty ``groups`` or zero layers) so callers fall back to
        single-group behavior. Registered tensors not referenced by any group
        are marked with ``EXCLUDED_ENGINE_GROUP`` (cross-layer KV-sharing layers
        whose KV lives in their target owner's blocks); downstream grouping
        skips them.

    Raises:
        ValueError: If a group references a layer index outside
            ``[0, num_registered_layers)``.
    """
    # First Party
    from lmcache.v1.kv_layer_groups import EXCLUDED_ENGINE_GROUP

    if not groups or num_registered_layers == 0:
        return None

    # Default to "excluded": layers no group references are intentionally left
    # out of grouping (e.g. KV-sharing layers aliasing a target owner's cache).
    per_layer_engine_group_idx = [EXCLUDED_ENGINE_GROUP] * num_registered_layers

    for group in groups:
        for layer_idx in group.layer_indices:
            if layer_idx < 0 or layer_idx >= num_registered_layers:
                raise ValueError(
                    f"Layer index {layer_idx} is outside registered layer "
                    f"range [0, {num_registered_layers})"
                )
            per_layer_engine_group_idx[layer_idx] = group.engine_group_id

    return per_layer_engine_group_idx
