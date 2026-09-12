# SPDX-License-Identifier: Apache-2.0
"""LookupModule: lookup, prefetch polling, and session lifecycle."""

# Standard
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    PrefetchMode,
    ObjectKey,
    PrefetchHandle,
    ipc_key_to_object_keys,
)
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import RestoreWindowResponse
from lmcache.v1.multiprocess.token_hasher import TokenHasher

if TYPE_CHECKING:
    # First Party
    from lmcache.native_storage_ops import Bitmap

logger = init_logger(__name__)


def compute_extra_count(
    tp_size: int,
    world_size: int,
    readers_per_object: int = 0,
) -> int:
    """Compute extra count for MLA multi-reader locking.

    New clients carry ``readers_per_object`` explicitly in the cache key.
    This is required for MLA+DCP: TP8/DCP4 stores four sequence-shard objects,
    and each object is read by two TP workers. The historical ``tp_size`` /
    ``world_size`` heuristic cannot distinguish that geometry.

    Non-MLA: each TP worker owns a distinct KV shard,
      so each ObjectKey is retrieved by exactly 1
      worker -> extra_count = 0.
    MLA: TP does not split KV caches, all TP workers
      share the same object. vLLM passes world_size
      already divided by tp_size (e.g. world_size=1
      for TP=4 PP=1), so ipc_keys_to_object_keys
      only produces 1 ObjectKey per chunk.  All TP
      workers retrieve that same ObjectKey, hence
      extra_count = tp_size - 1.

    Detection: tp > world_size means MLA (world_size
    was divided by tp on the vLLM side).

    Fallback: old vLLM (<= 0.8.5) does not send
    tp_size (defaults to 1); we fall back to
    world_size which gives extra_count = 0
    (safe but may under-lock for MLA).

    TODO: world_size currently carries an overloaded
    meaning (total ranks for non-MLA vs total/tp for
    MLA). Consider a dedicated field in the future.

    Args:
        tp_size: Tensor-parallel size from the client.
        world_size: World size from the cache key.
        readers_per_object: Explicit reader count, or 0 for the legacy
            heuristic.

    Returns:
        Number of extra count (0 for non-MLA).
    """
    if readers_per_object > 0:
        return readers_per_object - 1
    tp = tp_size if tp_size > 1 else world_size
    return tp - 1 if tp > world_size else 0


@dataclass
class _PrefetchJob:
    handles: tuple[PrefetchHandle, ...]
    world_size: int
    request_id: str
    # Number of tokens submitted for lookup (denominator for the L1+L2
    # token-level hit-rate metric).  Equals ``len(chunk_hashes) * chunk_size``
    # on the happy path; 0 for early-exit paths (no GPU context matches
    # or chunk_hashes is empty).  Consumed at ``MP_LOOKUP_PREFETCH_END``
    # emission time in ``query_prefetch_status``.
    requested_tokens: int
    object_keys_by_group: tuple[tuple[ObjectKey, ...], ...] = ()
    """Object keys aligned with ``handles``, in chunk/rank order per group."""
    prefetch_results: list["Bitmap | None"] = field(default_factory=list)
    """Completed result bitmaps retained across nonblocking status polls."""
    extra_count: int = 0
    """Additional read locks acquired for every prefetched object."""
    # Captured at lookup time so the ``MP_LOOKUP_PREFETCH_END`` event can
    # carry them as labels.  ``model_name`` lets dashboards slice hit rate
    # per model in multi-model deployments; ``cache_salt`` slices per
    # tenant / isolation domain (an empty string means no salt set).
    model_name: str = ""
    cache_salt: str = ""
    lookup_only: bool = False
    """True when no handle holds a lock: the job reported presence only, so
    completion must not release group-local surplus locks."""
    is_window: bool = False
    """True for a restore-window job: one rank's slice of a looked-up prefix,
    loaded on demand; its completion records no pinned prefix."""
    surplus_released: bool = False
    """Set once group-local surplus locks were released, so a job polled by
    several readers releases them exactly once."""

    def __post_init__(self) -> None:
        if not self.handles:
            raise ValueError("A prefetch job requires at least one handle")
        if not self.prefetch_results:
            self.prefetch_results = [None] * len(self.handles)
        elif len(self.prefetch_results) != len(self.handles):
            raise ValueError("Prefetch result slots must match the handle count")
        if self.object_keys_by_group and len(self.object_keys_by_group) != len(
            self.handles
        ):
            raise ValueError("Object-key groups must match the handle count")


class LookupModule:
    """Handles lookup, prefetch polling, lock release, and session lifecycle.

    Owns the prefetch-job bookkeeping (``_prefetch_jobs``) and exposes
    handlers for the LOOKUP, QUERY_PREFETCH_STATUS,
    QUERY_PREFETCH_LOOKUP_HITS, FREE_LOOKUP_LOCKS, and END_SESSION
    request types.

    Args:
        ctx: Shared engine context providing storage manager, token hasher,
            session manager, event bus, layout descriptor registry, and
            chunk size.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._prefetch_jobs: dict[str, _PrefetchJob] = {}
        self._prefetch_job_lock = threading.Lock()
        # Chunk count of the read-locked prefix of each looked-up request: the
        # L1 prefix hits plus the L2 head loaded under the pin limit. Chunks
        # past it were reported from the L2 index only and hold no lock; the
        # windowed restore loads them on demand. Recorded when the lookup
        # completes, dropped when the session ends.
        self._pinned_chunk_end: dict[str, int] = {}
        # Remaining readers of each restore-window job shared by workers that
        # read the same objects; the last reader consumes the job.
        self._window_job_readers: dict[str, int] = {}
        self._setup_metrics()

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves.

        Returns:
            List of handler specs for lookup-related request types.
        """
        return [
            HandlerSpec(RequestType.LOOKUP, self.lookup, ThreadPoolType.NORMAL),
            HandlerSpec(
                RequestType.QUERY_PREFETCH_STATUS,
                self.query_prefetch_status,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.WAIT_PREFETCH_STATUS,
                self.wait_prefetch_status,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.QUERY_PREFETCH_LOOKUP_HITS,
                self.query_prefetch_lookup_hits,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.FREE_LOOKUP_LOCKS,
                self.free_lookup_locks,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.RESTORE_WINDOW,
                self.restore_window,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.END_SESSION,
                self.end_session,
                ThreadPoolType.NORMAL,
            ),
        ]

    def report_status(self) -> dict[str, int]:
        """Return module-specific status information.

        Returns:
            Dictionary with the count of active prefetch jobs.
        """
        return {
            "active_prefetch_jobs": self._active_prefetch_count(),
        }

    def close(self) -> None:
        """Release resources owned by this module (no-op)."""
        pass

    # -----------------------------------------------------------------
    # Handlers
    # -----------------------------------------------------------------

    def lookup(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
    ) -> None:
        """Submit a prefix lookup.

        Hashes the key, submits a prefetch task to the storage manager,
        and registers the job under ``key.request_id`` for later polling
        via query_prefetch_status.

        Args:
            key: Cache key with request_id embedded.
            tp_size: Tensor-parallel size for MLA multi-reader locking.
        """
        model_name, world_size = key.model_name, key.world_size
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_START,
                session_id=key.request_id,
            )
        )
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id=key.request_id,
            )
        )

        layout_descs = self._ctx.layout_desc_registry.find_object_group_layouts(
            model_name, world_size
        )
        if layout_descs is None:
            logger.error(
                "No GPU context found for model %s with world size %d during lookup!",
                model_name,
                world_size,
            )
            self._register_prefetch_job(
                _PrefetchJob(
                    handles=(
                        PrefetchHandle(
                            prefetch_request_id=-1,
                            external_request_id=key.request_id,
                            l1_found_indices=(),
                            total_requested_keys=0,
                            submit_time=time.monotonic(),
                        ),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                    requested_tokens=0,
                    model_name=model_name,
                    cache_salt=key.cache_salt,
                )
            )
            return

        extra_count = compute_extra_count(tp_size, world_size, key.readers_per_object)

        chunk_hashes = self._ctx.token_hasher.compute_chunk_hashes(list(key.token_ids))
        if not chunk_hashes:
            self._register_prefetch_job(
                _PrefetchJob(
                    handles=(
                        PrefetchHandle(
                            prefetch_request_id=-1,
                            external_request_id=key.request_id,
                            l1_found_indices=(),
                            total_requested_keys=0,
                            submit_time=time.monotonic(),
                        ),
                    ),
                    world_size=1,
                    request_id=key.request_id,
                    requested_tokens=0,
                    model_name=model_name,
                    cache_salt=key.cache_salt,
                )
            )
            return

        # Total chunk-aligned tokens submitted for lookup; surfaces as the
        # denominator of the L1+L2 token-level hit-rate via the
        # ``requested_tokens`` field on ``MP_LOOKUP_PREFETCH_END``.  Sub-chunk
        # trailing tokens are intentionally excluded — they cannot hit at
        # chunk granularity.
        requested_tokens = len(chunk_hashes) * self._ctx.chunk_size

        # Guard with has_subscribers() to avoid allocating the metadata dict
        # (including dtype/shape list comprehensions) when no subscriber is
        # listening (e.g. lookup hash logger is disabled).
        if self._ctx.event_bus.has_subscribers(EventType.MP_LOOKUP):
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_LOOKUP,
                    session_id=key.request_id,
                    metadata={
                        "request_id": key.request_id,
                        "chunk_hashes": chunk_hashes,
                        "model_name": model_name,
                        "chunk_size": self._ctx.chunk_size,
                        "seq_len": len(key.token_ids),
                        "dtypes": [str(dtype) for dtype in layout_descs[0].dtypes],
                        "shapes": [list(shape) for shape in layout_descs[0].shapes],
                    },
                )
            )

        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        session.lookup_ipc_key = key

        attn_desc = self._ctx.layout_desc_registry.find_attn_desc(
            model_name, world_size
        )
        if len(layout_descs) != attn_desc.num_object_groups:
            raise ValueError(
                "Object-group layouts and attention windows must have the "
                f"same length: {len(layout_descs)} != {attn_desc.num_object_groups}"
            )
        object_keys_by_group = self._object_keys_by_group(key, chunk_hashes)
        # A lookup-only key reports the present prefix without loading or
        # locking: the client will not retrieve and will not free locks.
        mode = PrefetchMode.LOOKUP_ONLY if key.lookup_only else PrefetchMode.LOOKUP
        # Per group, keys are chunk-major over every rank, so a chunk pin limit
        # is ``chunks * ranks`` keys. Chunks past the limit are reported from
        # the L2 index and restored in windows by the workers.
        pin_limit_chunks = self._restore_pin_limit_chunks()
        pin_limit_keys = 0 if key.lookup_only else pin_limit_chunks * key.world_size
        handles = tuple(
            self._ctx.storage_manager.submit_prefetch_task(
                list(group_keys),
                layout_desc,
                extra_count=extra_count,
                external_request_id=key.request_id,
                attn_desc=AttnWindowDesc(
                    num_chunks_in_sw=[attn_desc.num_chunks_in_sw[object_group_id]]
                ),
                mode=mode,
                pin_limit_keys=pin_limit_keys,
            )
            for object_group_id, (group_keys, layout_desc) in enumerate(
                zip(object_keys_by_group, layout_descs, strict=True)
            )
        )
        self._register_prefetch_job(
            _PrefetchJob(
                handles=handles,
                world_size=key.world_size,
                request_id=key.request_id,
                requested_tokens=requested_tokens,
                object_keys_by_group=object_keys_by_group,
                extra_count=extra_count,
                model_name=model_name,
                cache_salt=key.cache_salt,
                lookup_only=key.lookup_only,
            )
        )

    def query_prefetch_lookup_hits(
        self,
        request_id: str,
    ) -> int | None:
        """Query the number of hits for a prefetch request before it's finished.

        Args:
            request_id: The external request ID passed in the lookup key.

        Returns:
            The number of hits for the prefetched keys if the lookup phase is
            done. None if the lookup phase is still in progress. 0 if the
            request_id is unknown (already completed and consumed, or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)

        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        found_prefix_lengths: list[int] = []
        for handle in job.handles:
            found = self._ctx.storage_manager.query_prefetch_lookup_hits(handle)
            if found is None:
                return None
            found_prefix_lengths.append(found)

        return min(found_prefix_lengths) // job.world_size

    def query_prefetch_status(
        self,
        request_id: str,
    ) -> int | None:
        """Poll the status of a prefetch job by request_id.

        Returns the chunk count when the prefetch is complete, or None
        if it is still in progress.  The job entry is automatically
        removed once a non-None result is returned (exactly-once
        semantics).

        Args:
            request_id: The external request ID passed in the lookup key.

        Returns:
            Chunk count (int) when done, None if still in progress,
            0 if the request_id is unknown (already completed and consumed,
            or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        found_by_group = self._query_prefetch_results(job)
        if found_by_group is None:
            return None

        found_count = min(
            found.count_leading_ones() // job.world_size for found in found_by_group
        )
        if not job.lookup_only and not job.surplus_released:
            job.surplus_released = True
            self._release_nonservable_group_results(job, found_by_group, found_count)
        if not job.lookup_only and not job.is_window:
            self._record_pinned_prefix(job, found_count)

        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id=job.request_id,
                metadata={
                    "found_count": found_count,
                    "requested_tokens": job.requested_tokens,
                    "hit_tokens": found_count * self._ctx.chunk_size,
                    "model_name": job.model_name,
                    "cache_salt": job.cache_salt,
                },
            )
        )

        self._consume_prefetch_job(request_id)

        return found_count

    def wait_prefetch_status(
        self,
        request_id: str,
        timeout: float,
    ) -> int | None:
        """Block until a prefetch job completes, then return its chunk count.

        Like query_prefetch_status, but waits for the daemon to publish the
        result instead of returning None while the prefetch is still in
        progress, so the caller does not have to busy-poll. The job entry is
        removed once a non-None result is returned (exactly-once semantics).

        Args:
            request_id: The external request ID passed in the lookup key.
            timeout: Maximum number of seconds to wait for the prefetch.

        Returns:
            Chunk count (int) when done, None if the wait timed out, 0 if the
            request_id is unknown (already completed and consumed, or invalid).
        """
        with self._prefetch_job_lock:
            job = self._prefetch_jobs.get(request_id)
        if job is None:
            logger.warning(
                "Prefetch job for request %s not found (already completed or invalid)",
                request_id,
            )
            return 0

        if len(job.handles) == 1:
            if not self._ctx.storage_manager.wait_prefetch_status(
                job.handles[0], timeout
            ):
                return None
        else:
            deadline = time.monotonic() + max(timeout, 0.0)
            for handle, result in zip(job.handles, job.prefetch_results, strict=True):
                if result is not None:
                    continue
                remaining = max(0.0, deadline - time.monotonic())
                if not self._ctx.storage_manager.wait_prefetch_status(
                    handle, remaining
                ):
                    return None
        return self.query_prefetch_status(request_id)

    def free_lookup_locks(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
    ) -> None:
        """Release read locks acquired during lookup.

        Hashes are computed only for chunks in ``[start, end)`` to avoid
        unnecessary work on tokens outside that range.
        ``start`` and ``end`` must be aligned to ``chunk_size``; it is the
        caller's responsibility to align the boundaries as desired.

        Computes the extra reader count from ``tp_size`` and
        ``world_size`` the same way :meth:`lookup` does, so
        the correct number of locks is released.

        Args:
            key: Cache key whose read locks should be released.
            tp_size: Tensor-parallel size for MLA
                multi-reader locking.
        """
        end = key.end
        if key.worker_id is None:
            # A lookup key names the whole looked-up prefix. Chunks past the
            # pinned prefix were reported from the L2 index and hold no lock;
            # releasing them would only log spurious lock errors.
            with self._prefetch_job_lock:
                pinned_chunk_end = self._pinned_chunk_end.get(key.request_id)
            if pinned_chunk_end is not None:
                end = min(end, pinned_chunk_end * self._ctx.chunk_size)
        # A worker key names a range this rank read-locked itself: a restore
        # window it loaded (``restore_window`` locks past the pinned prefix)
        # or the part of the pinned prefix it will not retrieve. Its bounds
        # are exact and must not be clipped, or the window locks leak and the
        # chunks can never be evicted.
        if end <= key.start:
            return
        chunk_hashes = self._ctx.token_hasher.compute_chunk_hashes(
            list(key.token_ids), start=key.start, end=end
        )
        if not chunk_hashes:
            return
        # Release across every object group, mirroring lookup, which locks keys
        # in every group; releasing only group 0 would leak the rest.
        #
        # NOTE: correct only for full attention, where every locked chunk is a
        # hit chunk. Sliding-window groups do not retain chunks outside their
        # window, so once SWA prefetch lands this must skip those chunks instead
        # of releasing every one -- otherwise chunks the engine still holds can
        # be over-released (e.g. window=512, LMCache hit 1024, vLLM hit 768 ->
        # chunks 512..768 may leak). Revisit when sliding-window prefetch is on.
        obj_keys = self._chunk_major_object_keys(key, chunk_hashes)

        extra_count = compute_extra_count(
            tp_size, key.world_size, key.readers_per_object
        )

        self._ctx.storage_manager.finish_read_prefetched(
            obj_keys, extra_count=extra_count
        )

    def end_session(self, request_id: str) -> None:
        """Remove the session for a finished request.

        Args:
            request_id: The request ID whose session should be removed.
        """
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_VLLM_END_SESSION,
                metadata={"request_id": request_id},
            )
        )
        session = self._ctx.session_manager.remove(request_id)
        with self._prefetch_job_lock:
            self._pinned_chunk_end.pop(request_id, None)
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_REQUEST_END,
                session_id=request_id,
            )
        )
        if session is None:
            logger.warning("Session %s not found, skipping touch", request_id)
            return
        if session.lookup_ipc_key is None:
            logger.warning(
                "Session %s has no lookup ipc key, skipping touch",
                request_id,
            )
            return

        chunk_hashes = [TokenHasher.hash_to_bytes(h) for h in session.get_hashes(0)]
        obj_keys = self._chunk_major_object_keys(session.lookup_ipc_key, chunk_hashes)
        # unified touch of all keys, which include retrieved and stored keys
        # TODO(chunxiaozheng): when l2 is enabled, the prefetched keys from l2 are temp
        #  and will be deleted after finish_read_prefetched, when we touch all keys,
        #  these keys has been deleted and will not be touched.
        self._ctx.storage_manager.touch_l1_keys(obj_keys)

    def restore_window(
        self,
        key: IPCCacheServerKey,
        tp_size: int,
    ) -> RestoreWindowResponse:
        """Load one window of a looked-up prefix from L2 into L1 for a worker.

        The lookup that preceded this call loaded and read-locked at most the
        server's pin limit of chunks and reported the rest from the L2 index.
        A worker restores that remainder window by window: this handler
        submits the window's objects of the worker's rank for loading and
        registers a prefetch job the worker waits on with
        WAIT_PREFETCH_STATUS before it retrieves the window. Chunks below the
        pinned prefix are skipped: they are already loaded and locked.

        Args:
            key: Worker key whose ``[start, end)`` token range is the window
                (chunk-aligned) and whose ``request_id`` is the looked-up
                request. ``worker_id`` must be set.
            tp_size: Tensor-parallel size, used with the key's reader count to
                take one read lock per retrieving worker.

        Returns:
            The window's loading plan; ``known=False`` when the request has no
            recorded lookup (its session ended), which tells the worker to stop.

        Raises:
            ValueError: If the key has no worker id or the range is not
                chunk-aligned.
        """
        if key.worker_id is None:
            raise ValueError("restore_window requires a worker key")
        chunk_size = self._ctx.chunk_size
        if key.start % chunk_size or key.end % chunk_size or key.end < key.start:
            raise ValueError(
                f"restore window [{key.start}, {key.end}) must align to "
                f"chunk size {chunk_size}"
            )
        with self._prefetch_job_lock:
            pinned_chunk_end = self._pinned_chunk_end.get(key.request_id)
        if pinned_chunk_end is None:
            return RestoreWindowResponse(known=False)

        load_start_chunk = max(key.start // chunk_size, pinned_chunk_end)
        end_chunk = key.end // chunk_size
        if load_start_chunk >= end_chunk:
            return RestoreWindowResponse(
                known=True, pinned_chunk_end=pinned_chunk_end, submitted_chunks=0
            )

        model_name, world_size = key.model_name, key.world_size
        layout_descs = self._ctx.layout_desc_registry.find_object_group_layouts(
            model_name, world_size
        )
        if layout_descs is None:
            raise ValueError(
                f"No GPU context found for model {model_name} with world size "
                f"{world_size} during restore_window"
            )
        attn_desc = self._ctx.layout_desc_registry.find_attn_desc(
            model_name, world_size
        )
        chunk_hashes = self._ctx.token_hasher.compute_chunk_hashes(
            list(key.token_ids), start=load_start_chunk * chunk_size, end=key.end
        )
        num_chunks = end_chunk - load_start_chunk
        if len(chunk_hashes) != num_chunks:
            raise ValueError(
                f"restore window [{key.start}, {key.end}) hashed "
                f"{len(chunk_hashes)} chunks, expected {num_chunks}"
            )
        object_keys_by_group = tuple(
            tuple(group_keys)
            for group_keys in ipc_key_to_object_keys(
                key, chunk_hashes, list(range(len(layout_descs)))
            )
        )
        extra_count = compute_extra_count(tp_size, world_size, key.readers_per_object)
        # Workers that read the same objects share one job (keyed by the
        # object rank); the lookup-style locking already takes one read lock
        # per reader, and the last reader consumes the job.
        kv_rank = object_keys_by_group[0][0].kv_rank
        job_id = f"{key.request_id}#w{load_start_chunk}#k{kv_rank}"
        with self._prefetch_job_lock:
            existing = self._prefetch_jobs.get(job_id)
        if existing is not None:
            return RestoreWindowResponse(
                known=True,
                pinned_chunk_end=pinned_chunk_end,
                submitted_chunks=num_chunks,
                job_id=job_id,
            )
        handles = tuple(
            self._ctx.storage_manager.submit_prefetch_task(
                list(group_keys),
                layout_desc,
                extra_count=extra_count,
                external_request_id=job_id,
                attn_desc=AttnWindowDesc(
                    num_chunks_in_sw=[attn_desc.num_chunks_in_sw[object_group_id]]
                ),
                mode=PrefetchMode.LOOKUP,
            )
            for object_group_id, (group_keys, layout_desc) in enumerate(
                zip(object_keys_by_group, layout_descs, strict=True)
            )
        )
        job = _PrefetchJob(
            handles=handles,
            world_size=1,
            request_id=job_id,
            requested_tokens=num_chunks * chunk_size,
            object_keys_by_group=object_keys_by_group,
            extra_count=extra_count,
            model_name=model_name,
            cache_salt=key.cache_salt,
            is_window=True,
        )
        with self._prefetch_job_lock:
            if job_id in self._prefetch_jobs:
                # Another reader registered the job first; theirs stands.
                self._window_job_readers[job_id] = 1 + extra_count
            else:
                self._prefetch_jobs[job_id] = job
                self._window_job_readers[job_id] = 1 + extra_count
        return RestoreWindowResponse(
            known=True,
            pinned_chunk_end=pinned_chunk_end,
            submitted_chunks=num_chunks,
            job_id=job_id,
        )

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _restore_pin_limit_chunks(self) -> int:
        """Chunks a lookup may load and read-lock before reporting the rest
        from the L2 index; 0 loads every found chunk."""
        return max(0, int(getattr(self._ctx, "restore_pin_limit_chunks", 0)))

    def _record_pinned_prefix(self, job: _PrefetchJob, found_count: int) -> None:
        """Remember how many leading chunks of a lookup hold read locks."""
        pinned = found_count
        for handle in job.handles:
            pinned_key_count = getattr(handle, "pinned_key_count", -1)
            if pinned_key_count >= 0:
                pinned = min(pinned, pinned_key_count // job.world_size)
        with self._prefetch_job_lock:
            self._pinned_chunk_end[job.request_id] = pinned

    def _consume_prefetch_job(self, request_id: str) -> None:
        """Drop a completed job; a shared window job waits for its last reader."""
        with self._prefetch_job_lock:
            readers = self._window_job_readers.get(request_id)
            if readers is not None and readers > 1:
                self._window_job_readers[request_id] = readers - 1
                return
            self._window_job_readers.pop(request_id, None)
            self._prefetch_jobs.pop(request_id, None)

    def _query_prefetch_results(self, job: _PrefetchJob) -> list["Bitmap"] | None:
        """Collect each object group's result without polling it twice."""
        for index, (handle, result) in enumerate(
            zip(job.handles, job.prefetch_results, strict=True)
        ):
            if result is not None:
                continue
            found = self._ctx.storage_manager.query_prefetch_status(handle)
            if found is not None:
                job.prefetch_results[index] = found

        if any(result is None for result in job.prefetch_results):
            return None
        return [result for result in job.prefetch_results if result is not None]

    def _release_nonservable_group_results(
        self,
        job: _PrefetchJob,
        found_by_group: list["Bitmap"],
        found_count: int,
    ) -> None:
        """Release group-local hits beyond the common model-wide prefix."""
        if not job.object_keys_by_group:
            return

        retained_keys_per_group = found_count * job.world_size
        surplus_keys: list[ObjectKey] = []
        for handle, group_keys, found in zip(
            job.handles, job.object_keys_by_group, found_by_group, strict=True
        ):
            # Keys at or past the handle's pinned count were reported from the
            # L2 index without a lock; only the locked surplus is released.
            pinned_key_count = getattr(handle, "pinned_key_count", -1)
            locked_end = pinned_key_count if pinned_key_count >= 0 else len(group_keys)
            surplus_keys.extend(
                group_keys[index]
                for index in found.get_indices_list()
                if retained_keys_per_group <= index < locked_end
            )
        if surplus_keys:
            self._ctx.storage_manager.finish_read_prefetched(
                surplus_keys, extra_count=job.extra_count
            )

    def _object_keys_by_group(
        self,
        key: IPCCacheServerKey,
        chunk_hashes: list[bytes],
    ) -> tuple[tuple[ObjectKey, ...], ...]:
        """Resolve chunk/rank-ordered object keys separately for each group."""
        num_groups = self._ctx.layout_desc_registry.find_attn_desc(
            key.model_name, key.world_size
        ).num_object_groups
        return tuple(
            tuple(group_keys)
            for group_keys in ipc_key_to_object_keys(
                key, chunk_hashes, list(range(num_groups))
            )
        )

    def _chunk_major_object_keys(
        self,
        key: IPCCacheServerKey,
        chunk_hashes: list[bytes],
    ) -> list[ObjectKey]:
        """Resolve the flat object-key list across all object groups,
        chunk-major.

        The object-group count is read from the layout registry for
        ``key``'s ``(model_name, world_size)``. The keys are ordered
        ``chunk -> object group -> kv_rank`` so that all keys belonging to one
        chunk are contiguous; a leading-ones prefix over the flat list then maps
        directly to a whole-chunk hit count. Callers that need the full key set
        regardless of order (lock release, touch) use this too.

        Example (2 chunks ``c0,c1``; 2 groups ``g0,g1``; 2 kv_ranks ``r0,r1``)::

            [c0g0r0, c0g0r1, c0g1r0, c0g1r1,   # chunk 0: all groups, all ranks
             c1g0r0, c1g0r1, c1g1r0, c1g1r1]   # chunk 1: ...

        Args:
            key: The IPC key (model/world/worker, salt).
            chunk_hashes: Chunk hashes to resolve keys for.

        Returns:
            The chunk-major flattened list of object keys across all groups.
        """
        per_group = self._object_keys_by_group(key, chunk_hashes)
        num_groups = len(per_group)
        if num_groups == 1:
            return list(per_group[0])
        # Each per-group list is chunk-major / rank-minor of length
        # len(chunk_hashes) * num_ranks; recover num_ranks to slice per chunk.
        num_ranks = len(per_group[0]) // len(chunk_hashes) if chunk_hashes else 0
        obj_keys: list[ObjectKey] = []
        for chunk_idx in range(len(chunk_hashes)):
            lo = chunk_idx * num_ranks
            hi = lo + num_ranks
            for group_keys in per_group:
                obj_keys.extend(group_keys[lo:hi])
        return obj_keys

    def _register_prefetch_job(self, job: _PrefetchJob) -> None:
        with self._prefetch_job_lock:
            self._prefetch_jobs[job.request_id] = job

    def _active_prefetch_count(self) -> int:
        """Return the number of active prefetch jobs (thread-safe)."""
        with self._prefetch_job_lock:
            return len(self._prefetch_jobs)

    def _setup_metrics(self) -> None:
        """Register OTel observable gauges for lookup module metrics."""
        _gauge = partial(register_gauge, "lmcache.mp_server")
        _gauge(
            "lmcache_mp.active_prefetch_jobs",
            "Number of active prefetch jobs",
            self._active_prefetch_count,
        )
