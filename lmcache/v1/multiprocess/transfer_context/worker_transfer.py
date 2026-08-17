# SPDX-License-Identifier: Apache-2.0
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Protocol
import os

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType, init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    GroupLayout,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import RegisterEngineDrivenContextResponse
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
    compute_kv_layout,
    create_engine_driven_context,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.transfer_context.pickle import EngineDrivenContextPickle
from lmcache.v1.platform import get_device_spec, resolve_kv_wrapper_factory
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.kv_wrap import wrap_kv_caches

if TYPE_CHECKING:
    # First Party
    import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)

# Environment variable that lets the user override the default routing
# performed by :func:`create_transfer_context`. Accepted values match the
# string values of :class:`MPTransferMode` (``auto`` / ``engine_driven`` /
# ``lmcache_driven``); ``auto`` reproduces the historical device-type-based
# dispatch.
ENV_MP_TRANSFER_MODE = "LMCACHE_MP_TRANSFER_MODE"


def _collapse_chunks_for_single_destination(
    chunks: list[torch.Tensor],
    block_ids: list[int],
    blocks_in_chunk: int,
    blocks_per_window: int,
) -> tuple[list[torch.Tensor], list[int]]:
    """Keep only the newest snapshot when every chunk aliases one destination.

    Mamba external hits contain null placeholders for historical states. The
    resulting block-ID list can be ``[0, 0, ..., 0]`` even though every LMCache
    object contains a different recurrent-state snapshot. Launching all objects
    together races multiple H2D writes to block zero. When every chunk maps to
    the same complete destination tuple, earlier writes are fully superseded,
    so only the newest chunk is semantically observable.

    Mixed or partial aliases are left unchanged; this helper only handles the
    proven full-alias case without changing unique full-attention transfers.
    """
    num_chunks = len(chunks)
    if (
        num_chunks <= 1
        or blocks_in_chunk != blocks_per_window
        or len(block_ids) != num_chunks * blocks_per_window
    ):
        return chunks, block_ids

    destinations = [
        tuple(block_ids[i * blocks_per_window : (i + 1) * blocks_per_window])
        for i in range(num_chunks)
    ]
    destination = destinations[0]
    if not destination or any(ids != destination for ids in destinations[1:]):
        return chunks, block_ids

    last_start = (num_chunks - 1) * blocks_per_window
    return chunks[-1:], block_ids[last_start : last_start + blocks_per_window]


# Helper functions
def _supports_async_primitives() -> bool:
    """Probe whether the worker device supports the async store primitives.

    The async engine-driven store path needs a stream, an event exposing
    ``record``/``synchronize``/``wait``, and pinned (page-locked) host memory.
    When any of these is unavailable (e.g. a CPU-only backend), the factory
    falls back to the synchronous :class:`EngineDrivenTransferContext`. This
    dispatch is internal and capability-based; there is no user-facing
    async/sync flag.

    Returns:
        True if all required async primitives are available, else False.
    """
    if not hasattr(torch_dev, "Stream") or not hasattr(torch_dev, "Event"):
        return False
    # CPU-only stub exposes Stream/Event but has no real async capability.
    if hasattr(torch_dev, "is_available") and not torch_dev.is_available():
        return False
    try:
        stream = torch_dev.Stream()
        event = torch_dev.Event()
    except Exception:
        return False
    for attr in ("record", "synchronize", "wait"):
        if not callable(getattr(event, attr, None)):
            del stream, event
            return False
    del stream, event
    try:
        probe = torch.empty(1, dtype=torch.uint8, device="cpu", pin_memory=True)
        del probe
    except (RuntimeError, TypeError):
        return False
    return True


def _build_engine_driven_context() -> "TransferContext":
    """Build the engine-driven context, async when device-capable else sync.

    Routes the ``ENGINE_DRIVEN`` and AUTO branches through a single capability
    check. ``AsyncEngineDrivenTransferContext`` is imported lazily to avoid an
    import cycle and to keep the synchronous path free of stream/event
    dependencies.

    Returns:
        ``AsyncEngineDrivenTransferContext`` when async primitives are
        available, otherwise ``EngineDrivenTransferContext``.
    """
    if _supports_async_primitives():
        # First Party
        from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
            AsyncEngineDrivenTransferContext,
        )

        logger.info("Using AsyncEngineDrivenTransferContext for store path")
        return AsyncEngineDrivenTransferContext()

    logger.info("Using EngineDrivenTransferContext (sync) for store path")
    return EngineDrivenTransferContext()


class MPTransferMode(str, Enum):
    """Routing mode used by :func:`create_transfer_context`.

    * ``AUTO``: dispatch by ``tensor.device.type`` (CUDA -> lmcache-driven,
      others -> engine-driven). Preserves the historical behaviour.
    * ``ENGINE_DRIVEN``: force :class:`EngineDrivenTransferContext`
      (worker-side gather / scatter copy path).
    * ``LMCACHE_DRIVEN``: force :class:`LMCacheDrivenTransferContext`
      (IPC / SHM zero-copy path). Requires a registered KV-wrapper factory
      for the device.
    """

    AUTO = "auto"
    ENGINE_DRIVEN = "engine_driven"
    LMCACHE_DRIVEN = "lmcache_driven"


def _resolve_mode(mode: "str | MPTransferMode | None") -> MPTransferMode:
    """Coerce ``mode`` into :class:`MPTransferMode`, falling back to env."""
    raw = (
        mode
        if mode is not None
        else os.environ.get(ENV_MP_TRANSFER_MODE, MPTransferMode.AUTO.value)
    )
    if isinstance(raw, MPTransferMode):
        return raw
    try:
        return MPTransferMode(str(raw).lower())
    except ValueError as exc:
        valid = ", ".join(m.value for m in MPTransferMode)
        raise ValueError(
            "Invalid MP transfer mode %r (valid: %s)" % (raw, valid)
        ) from exc


def _build_lmcache_driven_context(device_type: str) -> "TransferContext":
    """Build a :class:`LMCacheDrivenTransferContext` after capability check."""
    try:
        resolve_kv_wrapper_factory(device_type)
    except ValueError as exc:
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not supported for device type "
            "%r: no KV-cache wrapper factory is registered. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        ) from exc
    device_spec = get_device_spec(device_type)
    if device_spec and not device_spec.is_handle_transfer_available():
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not available for device type "
            "%r: required platform capability checks failed. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        )
    return LMCacheDrivenTransferContext()


class IPCEvent(Protocol):
    """Protocol for device events used by transport operations."""

    def wait(self, stream: object | None = None) -> None:
        """Make ``stream`` wait for this event (async ordering primitive)."""


SendRequest = Callable[[MessageQueueClient, RequestType, list[object]], MessagingFuture]


@dataclass
class _GroupState:
    """Worker-side per-LMCache-group transfer state (multi-group registration).

    Attributes:
        layer_names: KV cache dict keys belonging to this group, in group
            layer order — selects the gather/scatter tensor subset.
        engine_kv_format: Detected KV format for this group's tensors.
        blocks_in_chunk: Paged blocks of THIS group per LMCache chunk
            (``chunk_tokens / tokens_per_block``).
        blocks_per_window: Paged blocks of THIS group actually stored per
            chunk. Equals ``blocks_in_chunk`` for full-attention groups; for a
            sliding-window group it is ``window_tokens / tokens_per_block``, so
            only the trailing ``blocks_per_window`` blocks of each chunk are
            gathered / scattered.
        layout_desc: Chunk layout for this group's objects.
        logical_tokens_per_block: Global scheduler-token span represented by
            one group block ID.
        physical_tokens_per_block: Rank-local tensor slots held by that block.
        subblocks_per_manager: Number of fine external objects projected from
            one logical manager block.
    """

    layer_names: list[str]
    engine_kv_format: "lmc_ops.EngineKVFormat"
    blocks_in_chunk: int
    blocks_per_window: int
    layout_desc: MemoryLayoutDesc
    logical_tokens_per_block: int = 0
    physical_tokens_per_block: int = 0
    subblocks_per_manager: int = 1


def resolve_external_chunk_group_geometry(
    *,
    external_chunk_size: int,
    logical_tokens_per_block: int,
    physical_tokens_per_block: int,
    sliding_window_tokens: int,
) -> tuple[int, int, int, int]:
    """Resolve logical manager geometry into one rank-local object layout."""
    if (
        min(
            external_chunk_size,
            logical_tokens_per_block,
            physical_tokens_per_block,
        )
        <= 0
    ):
        raise ValueError("external, logical, and physical token sizes must be positive")

    if external_chunk_size < logical_tokens_per_block:
        if logical_tokens_per_block % external_chunk_size:
            raise ValueError(
                f"logical tokens_per_block {logical_tokens_per_block} is not a "
                f"multiple of external chunk size {external_chunk_size}"
            )
        if sliding_window_tokens >= 0:
            raise ValueError(
                "fine external chunks cannot split sliding-window or recurrent "
                "manager state"
            )
        subblocks_per_manager = logical_tokens_per_block // external_chunk_size
        if physical_tokens_per_block % subblocks_per_manager:
            raise ValueError(
                f"physical tokens_per_block {physical_tokens_per_block} cannot be "
                f"split into {subblocks_per_manager} external sub-blocks"
            )
        physical_window_tokens = physical_tokens_per_block // subblocks_per_manager
        return subblocks_per_manager, 1, 1, physical_window_tokens

    if external_chunk_size % logical_tokens_per_block:
        raise ValueError(
            f"external chunk size {external_chunk_size} is not a multiple of "
            f"logical tokens_per_block {logical_tokens_per_block}"
        )
    blocks_in_chunk = external_chunk_size // logical_tokens_per_block
    logical_window_tokens = (
        external_chunk_size
        if sliding_window_tokens < 0 or sliding_window_tokens >= external_chunk_size
        else sliding_window_tokens
    )
    if logical_window_tokens % logical_tokens_per_block:
        raise ValueError(
            f"sliding-window size {logical_window_tokens} is not a multiple of "
            f"logical tokens_per_block {logical_tokens_per_block}"
        )
    blocks_per_window = logical_window_tokens // logical_tokens_per_block
    return (
        1,
        blocks_in_chunk,
        blocks_per_window,
        blocks_per_window * physical_tokens_per_block,
    )


def make_external_subblock_view(
    kv_caches: dict[str, torch.Tensor],
    *,
    subblocks_per_manager: int,
    physical_tokens_per_block: int,
) -> dict[str, torch.Tensor]:
    """Return zero-copy blocks-first views split into external sub-blocks."""
    if subblocks_per_manager < 1:
        raise ValueError(
            f"subblocks_per_manager must be positive, got {subblocks_per_manager}"
        )
    if subblocks_per_manager == 1:
        return kv_caches
    if physical_tokens_per_block % subblocks_per_manager:
        raise ValueError(
            f"physical tokens_per_block {physical_tokens_per_block} cannot be "
            f"split into {subblocks_per_manager} external sub-blocks"
        )

    subblock_tokens = physical_tokens_per_block // subblocks_per_manager
    views: dict[str, torch.Tensor] = {}
    for layer_name, tensor in kv_caches.items():
        if (
            tensor.ndim != 3
            or tensor.shape[1] != physical_tokens_per_block
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "fine external chunks require contiguous one-plane blocks-first "
                "KV tensors shaped [num_blocks, block_size, hidden]; "
                f"layer {layer_name!r} has shape {tuple(tensor.shape)} and "
                f"contiguous={tensor.is_contiguous()}"
            )
        views[layer_name] = tensor.view(
            tensor.shape[0] * subblocks_per_manager,
            subblock_tokens,
            tensor.shape[2],
        )
    return views


def project_external_chunk_block_ids(
    block_ids: list[int],
    *,
    start_token_idx: int,
    external_chunk_size: int,
    logical_tokens_per_block: int,
    physical_tokens_per_block: int,
) -> tuple[list[int], int]:
    """Project manager IDs onto rank-local external sub-block IDs."""
    if (
        min(
            external_chunk_size,
            logical_tokens_per_block,
            physical_tokens_per_block,
        )
        <= 0
    ):
        raise ValueError("external, logical, and physical token sizes must be positive")
    if start_token_idx % external_chunk_size:
        raise ValueError(
            f"start token {start_token_idx} is not aligned to external chunk "
            f"size {external_chunk_size}"
        )
    if external_chunk_size >= logical_tokens_per_block:
        if external_chunk_size % logical_tokens_per_block:
            raise ValueError(
                f"external chunk size {external_chunk_size} is not a multiple "
                f"of logical tokens_per_block {logical_tokens_per_block}"
            )
        return list(block_ids), physical_tokens_per_block

    if logical_tokens_per_block % external_chunk_size:
        raise ValueError(
            f"logical tokens_per_block {logical_tokens_per_block} is not a "
            f"multiple of external chunk size {external_chunk_size}"
        )
    subblocks_per_manager = logical_tokens_per_block // external_chunk_size
    if physical_tokens_per_block % subblocks_per_manager:
        raise ValueError(
            f"physical tokens_per_block {physical_tokens_per_block} cannot be "
            f"split into {subblocks_per_manager} external sub-blocks"
        )
    first_external_chunk = start_token_idx // external_chunk_size
    previous_manager_id = block_ids[0] if block_ids else None
    for chunk_offset, manager_block_id in enumerate(block_ids[1:], start=1):
        subblock_idx = (first_external_chunk + chunk_offset) % subblocks_per_manager
        if subblock_idx and manager_block_id != previous_manager_id:
            raise ValueError(
                f"manager block ID changed before external sub-block wrap at "
                f"offset {chunk_offset}"
            )
        if not subblock_idx and manager_block_id == previous_manager_id:
            raise ValueError(
                f"manager block ID did not change at external sub-block wrap "
                f"at offset {chunk_offset}"
            )
        previous_manager_id = manager_block_id
    projected = [
        manager_block_id * subblocks_per_manager
        + (first_external_chunk + chunk_offset) % subblocks_per_manager
        for chunk_offset, manager_block_id in enumerate(block_ids)
    ]
    return projected, physical_tokens_per_block // subblocks_per_manager


def project_external_chunk_skip_tokens(
    logical_skip_tokens: int,
    *,
    logical_tokens_per_block: int,
    physical_tokens_per_block: int,
    subblocks_per_manager: int,
) -> int:
    """Convert a global-token skip to rows in a rank-local transfer view."""
    if subblocks_per_manager < 1:
        raise ValueError(
            f"subblocks_per_manager must be positive, got {subblocks_per_manager}"
        )
    if logical_skip_tokens < 0:
        raise ValueError(
            f"skip_first_n_tokens must be non-negative, got {logical_skip_tokens}"
        )
    if logical_tokens_per_block % subblocks_per_manager or (
        physical_tokens_per_block % subblocks_per_manager
    ):
        raise ValueError("logical and physical block sizes must split exactly")
    logical_chunk_span = logical_tokens_per_block // subblocks_per_manager
    if logical_skip_tokens % logical_chunk_span:
        raise ValueError(
            f"skip_first_n_tokens {logical_skip_tokens} must align to external "
            f"chunk span {logical_chunk_span}"
        )
    physical_chunk_span = physical_tokens_per_block // subblocks_per_manager
    return logical_skip_tokens // logical_chunk_span * physical_chunk_span


def _single_group_block_ids(block_ids: list[list[int]]) -> list[int]:
    """Return the flat block-id list for transports without HMA support."""
    if len(block_ids) != 1:
        raise RuntimeError(
            "engine-driven transfer does not support hybrid KV cache groups"
        )
    return block_ids[0]


def _get_kv_device(kv_caches: dict[str, torch.Tensor]) -> torch.device:
    """Return the device shared by a non-empty KV-cache mapping.

    Args:
        kv_caches: Worker KV-cache tensors keyed by layer name.

    Returns:
        The device of the first KV-cache tensor.

    Raises:
        ValueError: If ``kv_caches`` is empty.
    """
    if not kv_caches:
        raise ValueError("LMCache-driven transfer requires at least one KV cache")
    return next(iter(kv_caches.values())).device


class TransferContext(ABC):
    """Abstract transport layer for worker-side KV transfer.

    Concrete implementations encapsulate how worker-side store/retrieve
    operations are transmitted to the multiprocess server. Device-handle paths
    return event-aware futures backed by MQ requests, while CPU paths may perform
    gather/scatter synchronously and return already-resolved futures.
    """

    @abstractmethod
    def register(
        self,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register KV caches with the server and wait for ACK.

        Args:
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            model_name: Model name used by cache keys.
            world_size: KV world size.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            mq_client: Message queue client used to communicate with server.
            mq_timeout: Timeout in seconds for synchronous request wait.
            send_request: Request sender callable used to issue MQ requests.
            layout_hints: Optional inference-engine-provided layout hints.
            engine_group_infos: LMCache-owned engine KV cache group metadata.
            engine_type: Serving engine that produced the caches. Only
                consumed by the handle path; adapters should pass their
                own :class:`EngineType` so this transport stays engine-
                neutral. Defaults to :attr:`EngineType.VLLM` for
                backwards compatibility.

        Raises:
            TimeoutError: If server registration does not complete before
                ``mq_timeout``.
            RuntimeError: If a concrete context cannot initialize.
        """

    def register_q(
        self,
        instance_id: int,
        q_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        """Register the paged Q ring with the server under the same worker
        instance_id but different model_name (model_name##query).

        Args:
            instance_id: Worker process instance identifier.
            q_caches: Worker Q cache tensors keyed by layer name.
            model_name: Model name used by cache keys (model_name##query).
            world_size: KV world size.
            blocks_in_chunk: Number of Q ring blocks per LMCache chunk.
            mq_client: Message queue client used to communicate with server.
            mq_timeout: Timeout in seconds for synchronous request wait.
            send_request: Request sender callable used to issue MQ requests.
            layout_hints: Optional inference-engine-provided layout hints.
            engine_group_infos: LMCache-owned engine KV cache group metadata.

        Raises:
            NotImplementedError: If the concrete transport does not support the
                Q ring (now only lmcache-driven).
            TimeoutError: If server registration does not complete before
                ``mq_timeout``.
            RuntimeError: If a concrete context cannot initialize.
        """
        raise NotImplementedError(
            "Q ring registration is not supported by this transfer context"
        )

    def submit_q_store(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        q_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a Q ring store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key for the Q store range (query-specific model_name).
            instance_id: Worker process instance identifier (shared with KV).
            q_caches: Q ring tensors keyed by layer name.
            block_ids: Q ring block IDs to store, indexed by LMCache KV group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of Q ring blocks per LMCache chunk.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            NotImplementedError: If the concrete transport does not support the
                Q ring (only the lmcache-driven path does).
            RuntimeError: If register_q() was not called first.
        """
        raise NotImplementedError(
            "Q ring store is not supported by this transfer context"
        )

    @abstractmethod
    def submit_store(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the store range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to store, indexed by LMCache KV group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def submit_retrieve(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        """Submit a retrieve request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the retrieve range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to retrieve into, indexed by LMCache KV
                group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            skip_first_n_tokens: Number of initial tokens to skip when writing.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this context."""

    @abstractmethod
    def flush_inflight_stores(self) -> None:
        """Synchronize any in-flight gather operations.

        Subclasses must implement this method. Contexts with no deferred
        operations should implement it as a no-op. Async contexts that
        defer GPU->CPU gather work must block until all in-flight stores
        have completed, so that vLLM cannot overwrite paged KV blocks
        before they are read.
        """


class LMCacheDrivenTransferContext(TransferContext):
    """LMCache-driven IPC + MQ future transport context.

    In this mode the serving engine provides device handles (accelerator IPC,
    or SHM wrappers for CPU with IPC-like semantics) and the LMCache server
    performs direct device-side data transfer.
    """

    def __init__(self) -> None:
        self._mq_client: MessageQueueClient | None = None
        self._send_request: SendRequest | None = None
        self._device: torch.device | None = None
        self._event_backend: EventIPCBackend | None = None

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register the worker KV cache with the LMCache server.

        Args:
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV-cache tensors keyed by layer name.
            model_name: Model identifier used by the server.
            world_size: Tensor-parallel world size.
            _blocks_in_chunk: Engine blocks per LMCache chunk.
            mq_client: Message-queue client used for requests.
            mq_timeout: Timeout for the registration response.
            send_request: Request sender used by this context.
            layout_hints: Optional KV-layout metadata.
            engine_group_infos: Optional engine KV-group metadata.
            engine_type: Serving engine that produced the caches.

        Raises:
            RuntimeError: If event IPC is unsupported for the KV-cache device.
            ValueError: If ``kv_caches`` is empty.
        """
        device = _get_kv_device(kv_caches)
        event_backend = get_event_ipc_backend(device)
        event_backend.check_event_support(device)

        self._mq_client = mq_client
        self._send_request = send_request
        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE,
            [
                instance_id,
                wrap_kv_caches(kv_caches),
                model_name,
                world_size,
                engine_type,
                layout_hints,
                list(engine_group_infos),
            ],
        )
        future.result(timeout=mq_timeout)
        self._device = device
        self._event_backend = event_backend

    def register_q(
        self,
        instance_id: int,
        q_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        self._mq_client = mq_client
        self._send_request = send_request
        future = send_request(
            mq_client,
            RequestType.REGISTER_Q_CACHE,
            [
                instance_id,
                wrap_kv_caches(q_caches),
                model_name,
                world_size,
                EngineType.VLLM,
                layout_hints,
                list(engine_group_infos),
            ],
        )
        future.result(timeout=mq_timeout)

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a handle-based store ordered by ``event``.

        Args:
            _request_id: External request identifier (unused by this transport).
            key: LMCache key for the store range.
            instance_id: Worker process instance identifier.
            _kv_caches: Worker KV-cache tensors accepted for interface
                consistency; the registered device is reused.
            block_ids: Engine block IDs indexed by LMCache KV group.
            event: Producer event that orders reads of the engine KV cache.
            _blocks_in_chunk: Engine blocks per chunk (unused by this transport).

        Returns:
            A device-event-aware future for the server response.

        Raises:
            RuntimeError: If the context is not registered or event IPC is
                unsupported.
        """
        if (
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._send_request(
            self._mq_client,
            RequestType.STORE,
            [key, instance_id, block_ids, event_ipc_handle],
        ).to_device_future(device=self._device)

    def submit_q_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _q_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        if (
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_q_store()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._send_request(
            self._mq_client,
            RequestType.STORE_Q,
            [key, instance_id, block_ids, event_ipc_handle],
        ).to_device_future(device=self._device)

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        """Submit a handle-based retrieve ordered by ``event``.

        Args:
            _request_id: External request identifier (unused by this transport).
            key: LMCache key for the retrieve range.
            instance_id: Worker process instance identifier.
            _kv_caches: Worker KV-cache tensors accepted for interface
                consistency; the registered device is reused.
            block_ids: Engine block IDs indexed by LMCache KV group.
            event: Producer event that orders writes to the engine KV cache.
            _blocks_in_chunk: Engine blocks per chunk (unused by this transport).
            skip_first_n_tokens: Initial tokens the server must not overwrite.

        Returns:
            A device-event-aware future for the server response.

        Raises:
            RuntimeError: If the context is not registered or event IPC is
                unsupported.
        """
        if (
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._send_request(
            self._mq_client,
            RequestType.RETRIEVE,
            [key, instance_id, block_ids, event_ipc_handle, skip_first_n_tokens],
        ).to_device_future(device=self._device)

    def close(self) -> None:
        """Release the message queue and cached event-backend state."""
        self._mq_client = None
        self._send_request = None
        self._device = None
        self._event_backend = None

    def flush_inflight_stores(self) -> None:
        pass


class EngineDrivenTransferContext(TransferContext):
    """Engine-driven transfer context for non-CUDA workers.

    In this mode the engine (worker side) owns the data movement: the
    worker adapter gathers/packs KV into CPU buffers, commits via
    message-queue, and the server side persists/rehydrates from storage.
    """

    def __init__(self) -> None:
        self._engine_driven_context: EngineDrivenContext | None = None
        self._layout_hints: LayoutHints | None = None
        self._engine_kv_format: Any = None
        self._group_states: list[_GroupState] = []
        self._external_chunk_size = 0

    @property
    def engine_driven_context(self) -> EngineDrivenContext:
        """Return the underlying SHM/pickle context created by ``register``.

        Raises:
            RuntimeError: If accessed before ``register`` has run.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "EngineDrivenTransferContext is not registered, call register() first."
            )
        return self._engine_driven_context

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register KV caches with the non-GPU context server.

        With multiple ``engine_group_infos`` (hybrid-KV models), each group's
        layers are described by their own layout and gathered/scattered with
        that group's block-id list (uniform coverage: every group stores and
        retrieves every chunk; sliding-window groups keep only the trailing
        window of each chunk). Works over both transports: SHM gathers into
        server-reserved slots, pickle serializes a group-major payload.
        Single-group registration keeps the legacy path (see
        ``_single_group_block_ids``); fine sub-block geometry requires grouped
        object metadata and is rejected for a single explicit group.

        Logical tokens per block can exceed the physical slot dimension for
        context-parallel caches. Block selection remains in logical-token units,
        while payload shapes use rank-local physical slots.
        """
        del engine_type  # unused on the engine-driven path
        (
            block_size,
            num_layers,
            hidden_dim_size,
            dtype_str,
            engine_kv_format,
            kv_size,
        ) = compute_kv_layout(kv_caches, layout_hints=layout_hints)
        self._layout_hints = layout_hints
        self._engine_kv_format = engine_kv_format

        # The wire field is named use_mla but only drives the object plane
        # count: single-plane (kv_size == 1) covers MLA and fused-K/V formats.
        use_mla_flag = kv_size == 1
        chunk_tokens = blocks_in_chunk * block_size
        self._external_chunk_size = chunk_tokens

        group_layouts: list[GroupLayout] = []
        group_states: list[_GroupState] = []
        if len(engine_group_infos) == 1:
            logical_tokens_per_block = engine_group_infos[0].tokens_per_block
            if logical_tokens_per_block > 0 and chunk_tokens < logical_tokens_per_block:
                raise ValueError(
                    "single-group fine external chunks are unsupported; "
                    "register multiple object groups or use a chunk at least as "
                    "large as the logical manager block"
                )
        if len(engine_group_infos) > 1:
            layer_names = list(kv_caches)
            for gid, group in enumerate(engine_group_infos):
                subset = {
                    layer_names[i]: kv_caches[layer_names[i]]
                    for i in group.layer_indices
                }
                (
                    g_block_size,
                    g_num_layers,
                    g_hidden,
                    g_dtype_str,
                    g_format,
                    g_kv_size,
                ) = compute_kv_layout(subset, layout_hints=layout_hints)
                has_explicit_group_geometry = group.tokens_per_block > 0
                tokens_per_block = (
                    group.tokens_per_block
                    if has_explicit_group_geometry
                    else g_block_size
                )
                physical_block_size = g_block_size
                (
                    subblocks_per_manager,
                    group_blocks_in_chunk,
                    blocks_per_window,
                    physical_window_tokens,
                ) = resolve_external_chunk_group_geometry(
                    external_chunk_size=chunk_tokens,
                    logical_tokens_per_block=tokens_per_block,
                    physical_tokens_per_block=physical_block_size,
                    sliding_window_tokens=group.sw_size_tokens,
                )
                if subblocks_per_manager > 1:
                    make_external_subblock_view(
                        subset,
                        subblocks_per_manager=subblocks_per_manager,
                        physical_tokens_per_block=physical_block_size,
                    )
                g_mla = g_kv_size == 1
                g_shape = (
                    torch.Size([g_num_layers, physical_window_tokens, g_hidden])
                    if g_mla
                    else torch.Size([2, g_num_layers, physical_window_tokens, g_hidden])
                )
                group_layouts.append(
                    GroupLayout(
                        num_layers=g_num_layers,
                        hidden_dim_size=g_hidden,
                        dtype_str=g_dtype_str,
                        tokens_per_block=tokens_per_block,
                        window_tokens=physical_window_tokens,
                    )
                )
                group_states.append(
                    _GroupState(
                        layer_names=[layer_names[i] for i in group.layer_indices],
                        engine_kv_format=g_format,
                        blocks_in_chunk=group_blocks_in_chunk,
                        blocks_per_window=blocks_per_window,
                        layout_desc=MemoryLayoutDesc(
                            shapes=[g_shape],
                            dtypes=[getattr(torch, g_dtype_str)],
                        ),
                        logical_tokens_per_block=tokens_per_block,
                        physical_tokens_per_block=physical_block_size,
                        subblocks_per_manager=subblocks_per_manager,
                    )
                )
            # Group 0's layout doubles as the legacy top-level layout so
            # single-group readers of the payload keep working.
            shape = group_states[0].layout_desc.shapes[0]
            layout_desc = MemoryLayoutDesc(
                shapes=[shape], dtypes=group_states[0].layout_desc.dtypes
            )
        else:
            shape = (
                torch.Size([num_layers, chunk_tokens, hidden_dim_size])
                if use_mla_flag
                else torch.Size([2, num_layers, chunk_tokens, hidden_dim_size])
            )
            dtype = getattr(torch, dtype_str)
            layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
        self._group_states = group_states

        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
            [
                RegisterEngineDrivenContextPayload(
                    instance_id=instance_id,
                    model_name=model_name,
                    world_size=world_size,
                    block_size=block_size,
                    num_layers=num_layers,
                    hidden_dim_size=hidden_dim_size,
                    dtype_str=dtype_str,
                    use_mla=use_mla_flag,
                    group_layouts=group_layouts,
                )
            ],
        )
        response = future.result(timeout=mq_timeout)
        shm_name = ""
        pool_size = 0
        if isinstance(response, RegisterEngineDrivenContextResponse):
            shm_name = response.shm_name
            pool_size = response.pool_size

        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=block_size,
            use_mla=use_mla_flag,
        )
        self._engine_driven_context = create_engine_driven_context(
            metadata,
            mq_client,
            mq_timeout,
            shm_name=shm_name,
            pool_size=pool_size,
        )
        supported_transfer_mode = "SHM" if shm_name and pool_size > 0 else "pickle"
        logger.info(
            "Worker non-GPU transfer context registered "
            "(instance_id=%d, mode=%s, groups=%d)",
            instance_id,
            supported_transfer_mode,
            max(1, len(self._group_states)),
        )

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )

        if self._group_states:
            return self._submit_store_multigroup(key, instance_id, kv_caches, block_ids)

        torch_dev.synchronize()
        result = self._engine_driven_context.prepare_store(key, instance_id)
        out_buffers, chunk_indices = result if result is not None else (None, None)
        # All chunks already in cache — nothing to gather or commit.
        if chunk_indices is not None and len(chunk_indices) == 0:
            future: MessagingFuture[bool] = MessagingFuture()
            future.set_result(True)
            return future
        cpu_chunks = gather_paged_kv_to_cpu(
            kv_caches,
            _single_group_block_ids(block_ids),
            blocks_in_chunk,
            layout_hints=self._layout_hints,
            engine_kv_format=self._engine_kv_format,
            out=out_buffers,
            chunk_indices=chunk_indices,
        )
        # Gather issues async device->CPU copies on both transports (into SHM
        # slots or into fresh buffers that commit_store serializes); complete
        # them before commit either way.
        torch_dev.synchronize()
        ok = self._engine_driven_context.commit_store(key, instance_id, cpu_chunks)

        future = MessagingFuture()
        future.set_result(ok)
        return future

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )

        if self._group_states:
            return self._submit_retrieve_multigroup(
                key, instance_id, kv_caches, block_ids, skip_first_n_tokens
            )

        src_buffers = self._engine_driven_context.prepare_retrieve(key, instance_id)
        ok = src_buffers is not None
        if src_buffers is not None:
            try:
                scatter_cpu_to_paged_kv(
                    kv_caches,
                    _single_group_block_ids(block_ids),
                    src_buffers,
                    blocks_in_chunk,
                    skip_first_n_tokens=skip_first_n_tokens,
                    layout_hints=self._layout_hints,
                    engine_kv_format=self._engine_kv_format,
                )
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            # SHM path: ensure all device writes are complete before releasing
            # the SHM slot (server may immediately reuse it after commit_retrieve).
            torch_dev.synchronize()
        self._engine_driven_context.commit_retrieve(key, instance_id)

        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(ok)
        return future

    def _group_slots(
        self,
        tensors: list[torch.Tensor],
        per_slot_group_ids: list[int],
        gid: int,
        chunk_indices: "list[int] | None" = None,
    ) -> "tuple[list[torch.Tensor], list[int] | None]":
        """Select group ``gid``'s slot tensors (and chunk indices) in order."""
        idxs = [i for i, g in enumerate(per_slot_group_ids) if g == gid]
        picked = [tensors[i] for i in idxs]
        if chunk_indices is None:
            return picked, None
        return picked, [chunk_indices[i] for i in idxs]

    @staticmethod
    def _physical_skip_tokens(state: _GroupState, logical_skip_tokens: int) -> int:
        """Convert a global scheduler-token skip to rank-local tensor slots."""
        if state.logical_tokens_per_block <= 0 or state.physical_tokens_per_block <= 0:
            return logical_skip_tokens
        return project_external_chunk_skip_tokens(
            logical_skip_tokens,
            logical_tokens_per_block=state.logical_tokens_per_block,
            physical_tokens_per_block=state.physical_tokens_per_block,
            subblocks_per_manager=state.subblocks_per_manager,
        )

    def _group_transfer_inputs(
        self,
        state: _GroupState,
        key: Any,
        kv_caches: dict[str, torch.Tensor],
        manager_block_ids: list[int],
    ) -> tuple[dict[str, torch.Tensor], list[int]]:
        """Build zero-copy views and virtual IDs for one LMCache group."""
        group_kv_caches = {name: kv_caches[name] for name in state.layer_names}
        geometry_registered = (
            self._external_chunk_size > 0
            and state.logical_tokens_per_block > 0
            and state.physical_tokens_per_block > 0
        )
        if not geometry_registered:
            # Preserve the legacy coarse identity path for callers that predate
            # per-group geometry metadata. Fine projection must never use it.
            if (
                self._external_chunk_size != 0
                or state.logical_tokens_per_block != 0
                or state.physical_tokens_per_block != 0
                or state.subblocks_per_manager != 1
            ):
                raise RuntimeError("external chunk geometry was not fully registered")
            return group_kv_caches, list(manager_block_ids)

        token_span = key.end - key.start
        if token_span < 0 or token_span % self._external_chunk_size:
            raise ValueError(
                f"token range [{key.start}, {key.end}) does not align to "
                f"external chunk size {self._external_chunk_size}"
            )
        expected_block_ids = (
            token_span // self._external_chunk_size * state.blocks_in_chunk
        )
        if len(manager_block_ids) != expected_block_ids:
            raise ValueError(
                f"token range [{key.start}, {key.end}) requires "
                f"{expected_block_ids} block IDs, got {len(manager_block_ids)}"
            )
        if state.subblocks_per_manager == 1:
            return group_kv_caches, list(manager_block_ids)
        transfer_kv_caches = make_external_subblock_view(
            group_kv_caches,
            subblocks_per_manager=state.subblocks_per_manager,
            physical_tokens_per_block=state.physical_tokens_per_block,
        )
        transfer_block_ids, physical_tokens_per_chunk = (
            project_external_chunk_block_ids(
                manager_block_ids,
                start_token_idx=key.start,
                external_chunk_size=self._external_chunk_size,
                logical_tokens_per_block=state.logical_tokens_per_block,
                physical_tokens_per_block=state.physical_tokens_per_block,
            )
        )
        expected_physical_tokens = (
            state.physical_tokens_per_block // state.subblocks_per_manager
        )
        if physical_tokens_per_chunk != expected_physical_tokens:
            raise RuntimeError(
                "projected physical chunk span does not match registered view: "
                f"{physical_tokens_per_chunk} != {expected_physical_tokens}"
            )
        return transfer_kv_caches, transfer_block_ids

    def _submit_store_multigroup(
        self,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
    ) -> MessagingFuture:
        """Uniform-coverage store: gather every group's chunks into its slots."""
        ctx = self._engine_driven_context
        assert ctx is not None
        future: MessagingFuture[bool] = MessagingFuture()
        if len(block_ids) != len(self._group_states):
            raise RuntimeError(
                f"got {len(block_ids)} block-id lists for "
                f"{len(self._group_states)} registered groups"
            )
        if isinstance(ctx, EngineDrivenContextPickle):
            return self._submit_store_multigroup_pickle(
                ctx, key, instance_id, kv_caches, block_ids
            )
        # Validate and project every group's inputs before the SHM strategy
        # reserves write slots. A projection failure after prepare would leave
        # those slots pending because there is no store cancellation message.
        transfer_inputs = [
            self._group_transfer_inputs(state, key, kv_caches, block_ids[gid])
            for gid, state in enumerate(self._group_states)
        ]
        torch_dev.synchronize()
        result = ctx.prepare_store_grouped(key, instance_id)
        if result is None:
            future.set_result(False)
            return future
        tensors, chunk_indices, group_ids = result
        if not tensors:
            future.set_result(True)
            return future
        for gid, state in enumerate(self._group_states):
            out_g, chunks_g = self._group_slots(tensors, group_ids, gid, chunk_indices)
            if not out_g:
                continue
            transfer_kv_caches, transfer_block_ids = transfer_inputs[gid]
            gather_paged_kv_to_cpu(
                transfer_kv_caches,
                transfer_block_ids,
                state.blocks_in_chunk,
                layout_hints=self._layout_hints,
                engine_kv_format=state.engine_kv_format,
                out=out_g,
                chunk_indices=chunks_g,
                blocks_per_window=state.blocks_per_window,
            )
        # SHM writes are async device->CPU copies; complete them before commit.
        torch_dev.synchronize()
        ok = ctx.commit_store(key, instance_id, [])
        future.set_result(ok)
        return future

    def _submit_store_multigroup_pickle(
        self,
        ctx: "EngineDrivenContextPickle",
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
    ) -> MessagingFuture:
        """Uniform-coverage store over pickle: group-major serialized payload.

        There are no server-reserved slots in pickle mode, so each group's
        chunks are gathered into fresh CPU tensors and the whole
        ``chunks[group][chunk]`` list is pickled in one COMMIT_STORE payload.
        """
        future: MessagingFuture[bool] = MessagingFuture()
        torch_dev.synchronize()
        # Handshake only: the pickle strategy reserves nothing at prepare.
        ctx.prepare_store(key, instance_id)
        group_chunks: list[list[torch.Tensor]] = []
        for gid, state in enumerate(self._group_states):
            transfer_kv_caches, transfer_block_ids = self._group_transfer_inputs(
                state, key, kv_caches, block_ids[gid]
            )
            group_chunks.append(
                gather_paged_kv_to_cpu(
                    transfer_kv_caches,
                    transfer_block_ids,
                    state.blocks_in_chunk,
                    layout_hints=self._layout_hints,
                    engine_kv_format=state.engine_kv_format,
                    blocks_per_window=state.blocks_per_window,
                )
            )
        # Gather issues async device->CPU copies; commit_store serializes the
        # buffers immediately, so the copies must be complete first.
        torch_dev.synchronize()
        ok = ctx.commit_store(key, instance_id, group_chunks)
        future.set_result(ok)
        return future

    def _submit_retrieve_multigroup_pickle(
        self,
        ctx: "EngineDrivenContextPickle",
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        skip_first_n_tokens: int,
    ) -> MessagingFuture:
        """Uniform-coverage retrieve over pickle: scatter a group-major payload."""
        future: MessagingFuture[bool] = MessagingFuture()
        group_chunks = ctx.prepare_retrieve_multigroup(key, instance_id)
        if group_chunks is not None and len(group_chunks) != len(self._group_states):
            logger.error(
                "pickle retrieve returned %d groups for %d registered groups",
                len(group_chunks),
                len(self._group_states),
            )
            group_chunks = None
        ok = group_chunks is not None
        if group_chunks is not None:
            try:
                for gid, state in enumerate(self._group_states):
                    chunks = group_chunks[gid]
                    transfer_kv_caches, group_block_ids = self._group_transfer_inputs(
                        state, key, kv_caches, block_ids[gid]
                    )
                    if skip_first_n_tokens == 0:
                        compact_chunks, compact_block_ids = (
                            _collapse_chunks_for_single_destination(
                                chunks,
                                group_block_ids,
                                state.blocks_in_chunk,
                                state.blocks_per_window,
                            )
                        )
                        if len(compact_chunks) != len(chunks):
                            logger.debug(
                                "Collapsed %d aliased retrieve chunks to the "
                                "newest snapshot for engine group %d",
                                len(chunks),
                                gid,
                            )
                        chunks = compact_chunks
                        group_block_ids = compact_block_ids
                    scatter_cpu_to_paged_kv(
                        transfer_kv_caches,
                        group_block_ids,
                        chunks,
                        state.blocks_in_chunk,
                        skip_first_n_tokens=self._physical_skip_tokens(
                            state, skip_first_n_tokens
                        ),
                        layout_hints=self._layout_hints,
                        engine_kv_format=state.engine_kv_format,
                        blocks_per_window=state.blocks_per_window,
                    )
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            torch_dev.synchronize()
        ctx.commit_retrieve(key, instance_id)
        future.set_result(ok)
        return future

    def _submit_retrieve_multigroup(
        self,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        skip_first_n_tokens: int,
    ) -> MessagingFuture:
        """Uniform-coverage retrieve: scatter every group's chunks from its slots."""
        ctx = self._engine_driven_context
        assert ctx is not None
        future: MessagingFuture[bool] = MessagingFuture()
        if len(block_ids) != len(self._group_states):
            raise RuntimeError(
                f"got {len(block_ids)} block-id lists for "
                f"{len(self._group_states)} registered groups"
            )
        if isinstance(ctx, EngineDrivenContextPickle):
            return self._submit_retrieve_multigroup_pickle(
                ctx, key, instance_id, kv_caches, block_ids, skip_first_n_tokens
            )
        result = ctx.prepare_retrieve_grouped(key, instance_id)
        ok = result is not None
        if result is not None:
            tensors, group_ids = result
            try:
                for gid, state in enumerate(self._group_states):
                    src_g, _ = self._group_slots(tensors, group_ids, gid)
                    transfer_kv_caches, group_block_ids = self._group_transfer_inputs(
                        state, key, kv_caches, block_ids[gid]
                    )
                    if skip_first_n_tokens == 0:
                        src_g, group_block_ids = (
                            _collapse_chunks_for_single_destination(
                                src_g,
                                group_block_ids,
                                state.blocks_in_chunk,
                                state.blocks_per_window,
                            )
                        )
                    scatter_cpu_to_paged_kv(
                        transfer_kv_caches,
                        group_block_ids,
                        src_g,
                        state.blocks_in_chunk,
                        skip_first_n_tokens=self._physical_skip_tokens(
                            state, skip_first_n_tokens
                        ),
                        layout_hints=self._layout_hints,
                        engine_kv_format=state.engine_kv_format,
                        blocks_per_window=state.blocks_per_window,
                    )
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            # Complete device writes before the server may reuse the slots.
            torch_dev.synchronize()
        ctx.commit_retrieve(key, instance_id)
        future.set_result(ok)
        return future

    def close(self) -> None:
        if self._engine_driven_context is not None:
            self._engine_driven_context.close()
            self._engine_driven_context = None

    def flush_inflight_stores(self) -> None:
        pass


def create_transfer_context(
    kv_caches: dict[str, torch.Tensor],
    mode: "str | MPTransferMode | None" = None,
    **_kwargs: Any,
) -> TransferContext:
    """Create a transfer context from KV cache device type.

    The device check is intentionally centralized here. Routing can be
    overridden via the ``mode`` argument or the ``LMCACHE_MP_TRANSFER_MODE``
    environment variable; see :class:`MPTransferMode` for accepted values.

    Args:
        kv_caches: Worker KV cache tensors keyed by layer name.
        mode: Optional routing override. When ``None`` the value of
            ``LMCACHE_MP_TRANSFER_MODE`` is consulted, defaulting to
            :attr:`MPTransferMode.AUTO`.
        **kwargs: Unused placeholder for forward-compatible factory extension.

    Returns:
        A concrete :class:`TransferContext` implementation.

    Raises:
        ValueError: If ``kv_caches`` is empty, has mixed device types, the
            requested mode string is unknown, or the requested mode is not
            supported for the worker device.
    """
    if not kv_caches:
        raise ValueError("kv_caches is empty")
    device_types = {tensor.device.type for tensor in kv_caches.values()}
    if len(device_types) != 1:
        raise ValueError(
            f"All KV cache tensors must share one device type, got {device_types}"
        )
    device_type = next(iter(device_types))
    resolved_mode = _resolve_mode(mode)
    logger.info(
        "Creating transfer context (device_type=%s, mode=%s)",
        device_type,
        resolved_mode.value,
    )
    if resolved_mode is MPTransferMode.LMCACHE_DRIVEN:
        return _build_lmcache_driven_context(device_type)
    if resolved_mode is MPTransferMode.ENGINE_DRIVEN:
        return _build_engine_driven_context()
    # AUTO: dispatch by device type (CUDA -> handle path, else -> data path).
    if device_type == "cuda":
        return LMCacheDrivenTransferContext()
    return _build_engine_driven_context()
