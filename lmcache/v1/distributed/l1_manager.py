# SPDX-License-Identifier: Apache-2.0
"""
Managing objects and memory for L1 cache
"""

# Standard
from dataclasses import dataclass
from itertools import chain, islice
from typing import Literal
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import TTLLock
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.distributed.memory_manager import (
    GDSL1MemoryManager,
    L1ManagerProtocol,
    L1MemoryManager,
)
from lmcache.v1.distributed.memory_manager.devdax_l1_memory_manager import (
    DevDaxL1MemoryManager,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge

logger = init_logger(__name__)


# Internal classes and helper functions
@dataclass
class L1ObjectState:
    """
    The internal state of an object in L1 cache
    """

    memory_obj: MemoryObj
    """ The memory object stored in L1 cache. """

    write_lock: TTLLock
    """ Whether the object is write-locked. """

    read_lock: TTLLock
    """ The read lock with TTL for the object. """

    is_temporary: bool
    """ Whether the object is temporary (need to be deleted after read). """

    write_pending: bool = True
    """ True from ``reserve_write`` until the writer finishes the write.

    The write lock's TTL only bounds how long a writer may hold the object;
    when it expires the object becomes reclaimable, not readable. Without
    this flag an object whose writer died (or never committed) would be
    served as a cache hit with uninitialized contents once the TTL ran out,
    and the store controller would persist it to L2. """

    def available_for_read(self) -> bool:
        """Check if the object is available for read.

        Returns:
            True if the object's write has been finished and it is not
            write-locked, False otherwise. An object whose write lock
            expired before ``finish_write`` stays unreadable.
        """
        return not self.write_pending and not self.write_lock.is_locked()

    def available_for_write(self) -> bool:
        """Check if the object is available for write.

        Returns:
            True if the object is not write-locked and has no read locks
            and is not a temporary object, False otherwise.
        """

        return (
            not self.write_lock.is_locked()
            and not self.read_lock.is_locked()
            and not self.is_temporary
        )


def l1_mgr_synchronized(func):
    """
    Decorator to mark L1Manager methods as thread-safe
    """

    def wrapper(self: "L1Manager", *args, **kwargs):
        with self._lock:
            return func(self, *args, **kwargs)

    return wrapper


L1OperationResult = tuple[L1Error, MemoryObj | None]

# Upper bound for the count parameter in reserve_read / finish_read
# to prevent a single call from holding the global lock for too long.
MAX_READ_LOCK_COUNT = 128


def _validate_extra_count(extra_count: int) -> int:
    """Validate and clamp extra_count.

    Args:
        extra_count: Extra lock count on top of the
            default 1 lock.

    Returns:
        Clamped value in [0, MAX_READ_LOCK_COUNT - 1].
    """
    if extra_count < 0:
        logger.warning(
            "L1Manager: extra_count=%d is invalid, clamping to 0",
            extra_count,
        )
        return 0
    upper = MAX_READ_LOCK_COUNT - 1
    if extra_count > upper:
        logger.warning(
            "L1Manager: extra_count=%d exceeds limit=%d, clamping",
            extra_count,
            upper,
        )
        return upper
    return extra_count


def _l1_usage_ratio_or_zero(target: "L1Manager | None") -> float:
    """Return ``target.get_memory_usage()`` as a 0.0-1.0 ratio.

    Returns 0.0 when ``target`` is None or ``total_bytes`` is zero so the
    observable-gauge callback never raises during scrape.
    """
    if target is None:
        return 0.0
    used, total = target.get_memory_usage()
    if total <= 0:
        return 0.0
    return used / total


# Main classes


class L1Manager:
    """
    Object lifecycle state machine for L1 cache

          +--------+
          |  None  | <---------------------------------------+
          +--------+                                         |
            |   ^                                            |
            |   | (write lock expired)                       | delete()
            |   |                                            |
    reserve |   +----------------------+                     |
    write() |                          |                     |
            v                          |                     |
      +--------------+           +-----------+               |
      | write_locked |           |           |---------------+
      |              |---------->|   ready   |
      |              | finish_   |           |---------------+
      +--------------+ write()   +-----------+               |
            ^                          |                     |
            |                          | reserve_read()      | finish_read()
            +--------------------------+                     | (if count becomes 0)
                 reserve_write()       |                     |
                                       v                     |
                               +-----------------+           |
                               |   read_locked   |-----------+
                               |   (count = 1)   |
                               +-----------------+
                                     |     ^
                      reserve_read() |     | finish_read()
                                     v     |
                               +-----------------+
                               |   read_locked   |
                               |   (count = 2)   |
                               +-----------------+
                                     |     ^
                      reserve_read() |     | finish_read()
                                     v     |
                                   (...)  (...)
                               (Higher Counts)

    For every operation on list of keys, the operation is atomic
    """

    # Singleton dispatch for ``lmcache_mp.l1_memory_usage_bytes``: tests may
    # construct multiple L1Managers but the OTel SDK only honors the first
    # gauge registration, so the callback reads from the most recently built
    # instance via ``_gauge_target``.
    _gauge_registered: bool = False
    _gauge_target: "L1Manager | None" = None

    def __init__(self, config: L1ManagerConfig):
        self._lock = threading.Lock()

        self._objects: dict[ObjectKey, L1ObjectState] = {}

        # GDS, Device-DAX, and CPU L1 are mutually exclusive tiers. Each tier
        # owns its backing allocator instead of branching inside the CPU path.
        self._memory_manager: L1ManagerProtocol
        if config.gds_l1_config is not None:
            self._memory_manager = GDSL1MemoryManager(config.gds_l1_config)
            logger.info("L1Manager: GDS L1 tier enabled; CPU pinned-DRAM L1 disabled")
        elif config.memory_config.devdax_path:
            self._memory_manager = DevDaxL1MemoryManager(config.memory_config)
            logger.info("L1Manager: Device-DAX L1 tier enabled; CPU-only L1 disabled")
        else:
            self._memory_manager = L1MemoryManager(config.memory_config)

        self._write_ttl_seconds = config.write_ttl_seconds
        self._read_ttl_seconds = config.read_ttl_seconds

        self._registered_listeners: list[L1ManagerListener] = []

        self._event_bus = get_event_bus()

        L1Manager._gauge_target = self
        if not L1Manager._gauge_registered:
            L1Manager._gauge_registered = True
            register_gauge(
                "lmcache.l1_manager",
                "lmcache_mp.l1_memory_usage_bytes",
                "Bytes currently held in L1 cache",
                lambda: (
                    L1Manager._gauge_target.get_memory_usage()[0]
                    if L1Manager._gauge_target is not None
                    else 0
                ),
            )
            register_gauge(
                "lmcache.l1_manager",
                "lmcache_mp.l1_usage_ratio",
                "L1 used/total ratio (0.0–1.0)",
                lambda: _l1_usage_ratio_or_zero(L1Manager._gauge_target),
            )

    def register_listener(self, listener: L1ManagerListener) -> None:
        """Register a listener for L1Manager events.

        Args:
            listener: The listener to register.
        """
        with self._lock:
            self._registered_listeners.append(listener)

    @l1_mgr_synchronized
    def reserve_read(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve read access for the given keys.

        Args:
            keys: The list of object keys to reserve
                read access for.
            extra_count: Extra read locks on top of the
                default 1 lock.  Total locks acquired per
                key = 1 + extra_count.  Useful when multiple
                workers each consume one read lock for the
                same key (e.g. MLA models with TP > 1).

        Returns:
            A dictionary mapping each object key to a tuple
            of (L1Error, Optional[MemoryObj]).

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_NOT_READABLE: The key exists but is not
                readable.
        """
        extra_count = _validate_extra_count(extra_count)
        total = 1 + extra_count
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []
        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = (L1Error.KEY_NOT_EXIST, None)
                continue

            if not entry.available_for_read():
                ret[key] = (L1Error.KEY_NOT_READABLE, None)
                continue

            # TODO(perf): support a count argument in
            # TTLLock.lock() to avoid Python for-loop
            # overhead (TTLLock is C++ std::atomic).
            for _ in range(total):
                entry.read_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            successful_keys.append(key)

        for listener in self._registered_listeners:
            listener.on_l1_keys_reserved_read(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_RESERVED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    @l1_mgr_synchronized
    def unsafe_read(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, L1OperationResult]:
        """Unsafe read the read-locked objects without adding new read locks.

        This method does not acquire read locks. Therefore, the caller need
        to make sure the `unsafe_read` is called between `reserve_read` and
        `finish_read` calls.

        Args:
            keys: The list of object keys to read.

        Returns:
            A dictionary mapping each object key to a tuple of
            (L1Error, Optional[MemoryObj]).

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_NOT_READABLE: The key is not readable (in this case, not read-locked).
        """
        ret: dict[ObjectKey, L1OperationResult] = {}

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = (L1Error.KEY_NOT_EXIST, None)
                continue

            if not entry.read_lock.is_locked():
                ret[key] = (L1Error.KEY_NOT_READABLE, None)
                continue

            ret[key] = (L1Error.SUCCESS, entry.memory_obj)

        return ret

    @l1_mgr_synchronized
    def finish_read(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> dict[ObjectKey, L1Error]:
        """Finish read access for the given keys.

        Will delete the object if it is temporary and read
        count reaches zero.

        Args:
            keys: The list of object keys to finish read
                access for.
            extra_count: Extra read locks to release on top
                of the default 1.  Must match the
                ``extra_count`` used in the corresponding
                ``reserve_read`` call.

        Returns:
            A dictionary mapping each object key to an
            L1Error.

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IN_WRONG_STATE: The key is write-locked or
                non-read-locked, which means the reader may
                read inconsistent data.
        """
        extra_count = _validate_extra_count(extra_count)
        total = 1 + extra_count
        need_to_free: list[MemoryObj] = []
        need_to_free_keys: list[ObjectKey] = []
        ret: dict[ObjectKey, L1Error] = {}
        successful_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                logger.warning(
                    "L1Manager: finish read on non-existing key %s, "
                    "potential inconsistent data might be read",
                    key,
                )
                ret[key] = L1Error.KEY_NOT_EXIST
                continue

            if entry.write_lock.is_locked():
                logger.warning(
                    "L1Manager: finish read on write-locked key %s, "
                    "potential inconsistent data might be read",
                    key,
                )
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue

            if not entry.read_lock.is_locked():
                logger.warning(
                    "L1Manager: finish read on non-read-locked key %s, "
                    "potential inconsistent data might be read",
                    key,
                )
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue

            # TODO(perf): support a count argument in
            # TTLLock.unlock() to avoid Python for-loop
            # overhead (TTLLock is C++ std::atomic).
            for _ in range(total):
                entry.read_lock.unlock()
            if entry.is_temporary and not entry.read_lock.is_locked():
                # NOTE: temporary objects shouldn't have write-locks
                need_to_free.append(entry.memory_obj)
                need_to_free_keys.append(key)
                del self._objects[key]

            ret[key] = L1Error.SUCCESS
            successful_keys.append(key)

        self._memory_manager.free(need_to_free)

        for listener in self._registered_listeners:
            listener.on_l1_keys_read_finished(successful_keys)
            listener.on_l1_keys_deleted_by_manager(need_to_free_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_READ_FINISHED,
                metadata={"keys": successful_keys},
            )
        )
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                metadata={"keys": need_to_free_keys},
            )
        )

        return ret

    @l1_mgr_synchronized
    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        mode: Literal["new", "update", "all"] = "all",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve write access for the given keys.

        Args:
            keys: The list of object keys to reserve write access for.
            is_temporary: The list of booleans indicating whether each key is
                temporary.
            shape_spec: The memory layout description for the objects to be
                allocated.
            mode (Literal["new", "update", "all"]): Reservation mode.
            - "new": Reserve only new objects that do not exist.
            - "update": Reserve only existing objects for update.
            - "all": Reserve all writable objects regardless of existence.

        Returns:
            A dictionary mapping each object key to a tuple of
            (L1Error, Optional[MemoryObj]).

        Errors:
            KEY_NOT_WRITABLE: The key exists but is not writable.
            OUT_OF_MEMORY: Not enough memory to allocate for the object.

        Note:
            When the full batch does not fit, the longest prefix of the
            to-be-allocated keys that fits is allocated and the remaining
            keys report OUT_OF_MEMORY.
        """
        need_to_allocate: list[tuple[ObjectKey, bool]] = []
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []

        for key, is_temp in zip(keys, is_temporary, strict=False):
            entry = self._objects.get(key, None)
            if entry is None:
                need_to_allocate.append((key, is_temp))
                continue

            if mode == "new":
                ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                continue

            if not entry.available_for_write():
                ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                continue

            entry.write_lock.lock()
            entry.write_pending = True
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            successful_keys.append(key)

        # Early return if no allocation is needed
        if len(need_to_allocate) == 0:
            return ret

        # Don't allow allocation in "update" mode
        if mode == "update":
            for key, _ in need_to_allocate:
                ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
            return ret

        err, allocated_objs = self._memory_manager.allocate(
            layout_desc, len(need_to_allocate)
        )

        # A full-batch allocation failure would fail every key even when
        # most of the batch fits, collapsing large L2->L1 restores to zero
        # prefix hits and forcing full re-prefills upstream. Fall back to
        # allocating the longest prefix that fits; callers already handle
        # per-key OOM (prefetch trims to the contiguous prefix, the store
        # path skips failed keys).
        if err != L1Error.SUCCESS and len(need_to_allocate) > 1:
            if allocated_objs:
                self._memory_manager.free(allocated_objs)
                allocated_objs = []
            err, allocated_objs = self._allocate_largest_prefix(
                layout_desc, len(need_to_allocate)
            )
            if err == L1Error.SUCCESS:
                logger.warning(
                    "Partial L1 allocation: %d/%d objects (prefix) allocated",
                    len(allocated_objs),
                    len(need_to_allocate),
                )

        if err != L1Error.SUCCESS:
            for key, _ in need_to_allocate:
                ret[key] = (L1Error.OUT_OF_MEMORY, None)

            # Free the memory if partial allocation succeeded
            if allocated_objs:
                self._memory_manager.free(allocated_objs)

        else:
            # Mark any unallocated tail as OOM so every requested key keeps
            # an explicit per-key result.
            for key, _ in need_to_allocate[len(allocated_objs) :]:
                ret[key] = (L1Error.OUT_OF_MEMORY, None)
            for (key, is_temp), mem_obj in zip(
                need_to_allocate, allocated_objs, strict=False
            ):
                self._objects[key] = L1ObjectState(
                    memory_obj=mem_obj,
                    write_lock=TTLLock(self._write_ttl_seconds),
                    read_lock=TTLLock(self._read_ttl_seconds),
                    is_temporary=is_temp,
                )
                self._objects[key].write_lock.lock()
                ret[key] = (L1Error.SUCCESS, mem_obj)
                successful_keys.append(key)

        for listener in self._registered_listeners:
            listener.on_l1_keys_reserved_write(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_RESERVED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    def _allocate_largest_prefix(
        self, layout_desc: MemoryLayoutDesc, want: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Allocate the largest prefix below an already-failed full batch.

        L1 allocators have an all-or-nothing batch contract. While holding the
        L1 manager lock, allocation feasibility is therefore monotonic: if a
        batch of size ``n`` fits, every smaller batch also fits. Binary search
        finds the maximum without relying on byte estimates that can be wrong
        for alignment or fragmented free lists. Successful probes are freed
        before the final allocation.

        This recovery path runs only after the ``want`` allocation failed, so
        normal reservations still perform exactly one allocation.
        """
        low = 1
        high = want - 1
        largest = 0

        while low <= high:
            candidate = (low + high) // 2
            err, objects = self._memory_manager.allocate(layout_desc, candidate)
            if err == L1Error.SUCCESS:
                largest = candidate
                self._memory_manager.free(objects)
                low = candidate + 1
            else:
                if objects:
                    self._memory_manager.free(objects)
                high = candidate - 1

        if largest == 0:
            return L1Error.OUT_OF_MEMORY, []
        return self._memory_manager.allocate(layout_desc, largest)

    @l1_mgr_synchronized
    def finish_write(
        self,
        keys: list[ObjectKey],
    ) -> dict[ObjectKey, L1Error]:
        """Finish write access for the given keys.

        Args:
            keys: The list of object keys to finish write access for.

        Returns:
            A dictionary mapping each object key to an L1Error.

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IN_WRONG_STATE: The key is not write-locked, or it's read-locked,
                which means the writer may have caused inconsistent data.
        """
        ret: dict[ObjectKey, L1Error] = {}
        successful_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = L1Error.KEY_NOT_EXIST
                continue

            if not entry.write_lock.is_locked():
                logger.warning(
                    "L1Manager: finish write on non-write-locked key %s, "
                    "potential inconsistent data might be written",
                    key,
                )
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue

            if entry.read_lock.is_locked():
                logger.warning(
                    "L1Manager: finish write on read-locked key %s, "
                    "potential inconsistent data might be written",
                    key,
                )
                ret[key] = L1Error.KEY_IN_WRONG_STATE
                continue

            entry.write_lock.unlock()
            entry.write_pending = False
            ret[key] = L1Error.SUCCESS
            successful_keys.append(key)

        for listener in self._registered_listeners:
            listener.on_l1_keys_write_finished(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_FINISHED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    @l1_mgr_synchronized
    def finish_write_and_reserve_read(
        self,
        keys: list[ObjectKey],
        extra_count: int = 0,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Atomically finish write and acquire read lock for the given keys.

        This is used by the prefetch controller after successfully loading
        data from L2 into write-reserved L1 buffers. It transitions the
        object from write-locked to read-locked in a single atomic step,
        preventing a race window where eviction could interfere.

        Args:
            keys: Keys to transition from write-locked to read-locked.
            extra_count: Extra read locks on top of the default 1 lock.
                Total locks acquired per key = 1 + extra_count.  Useful
                when multiple TP workers each consume one read lock for
                the same key (e.g. MLA models with TP > 1).

        Returns:
            A dictionary mapping each object key to a tuple of
            (L1Error, Optional[MemoryObj]).

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IN_WRONG_STATE: The key is not write-locked, or it already
                has read locks.
        """
        extra_count = _validate_extra_count(extra_count)
        total = 1 + extra_count
        ret: dict[ObjectKey, L1OperationResult] = {}
        successful_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = (L1Error.KEY_NOT_EXIST, None)
                continue

            if not entry.write_lock.is_locked():
                logger.warning(
                    "L1Manager: finish_write_and_reserve_read on "
                    "non-write-locked key %s",
                    key,
                )
                ret[key] = (L1Error.KEY_IN_WRONG_STATE, None)
                continue

            if entry.read_lock.is_locked():
                logger.warning(
                    "L1Manager: finish_write_and_reserve_read on read-locked key %s",
                    key,
                )
                ret[key] = (L1Error.KEY_IN_WRONG_STATE, None)
                continue

            entry.write_lock.unlock()
            entry.write_pending = False
            for _ in range(total):
                entry.read_lock.lock()
            ret[key] = (L1Error.SUCCESS, entry.memory_obj)
            successful_keys.append(key)

        for listener in self._registered_listeners:
            listener.on_l1_keys_finish_write_and_reserve_read(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    @l1_mgr_synchronized
    def discard_abandoned_writes(self, keys: list[ObjectKey]) -> list[ObjectKey]:
        """Delete objects whose writer reserved them and never finished.

        An object is abandoned when ``write_pending`` is still set and its
        write lock TTL has expired, i.e. the writer died or gave up before
        ``finish_write``. Such objects hold uninitialized memory, can never
        become readable, and would otherwise stay resident until the next
        restart because the write-back eviction path only deletes objects it
        could read and persist.

        Args:
            keys: Candidate keys; keys that are absent, readable, still
                write-locked or read-locked are left untouched.

        Returns:
            The keys that were deleted.
        """
        need_to_free: list[MemoryObj] = []
        discarded: list[ObjectKey] = []
        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None or not entry.write_pending:
                continue
            if entry.write_lock.is_locked() or entry.read_lock.is_locked():
                continue
            need_to_free.append(entry.memory_obj)
            del self._objects[key]
            discarded.append(key)

        if not discarded:
            return discarded
        logger.warning(
            "L1Manager: discarding %d abandoned write reservations "
            "(write lock expired before finish_write)",
            len(discarded),
        )
        self._memory_manager.free(need_to_free)
        for listener in self._registered_listeners:
            listener.on_l1_keys_deleted_by_manager(discarded)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                metadata={"keys": discarded},
            )
        )
        return discarded

    @l1_mgr_synchronized
    def delete(
        self, keys: list[ObjectKey], force: bool = False
    ) -> dict[ObjectKey, L1Error]:
        """Delete the given keys from L1 cache.

        Args:
            keys: The list of object keys to delete.
            force: When True, delete even a read/write-locked key. This may free
                memory a concurrent store/read still uses (same hazard as
                :meth:`clear` with ``force=True``); use with care.

        Returns:
            A dictionary mapping each object key to an L1Error.

        Errors:
            KEY_NOT_EXIST: The key does not exist.
            KEY_IS_LOCKED: The key is write-locked or read-locked and cannot be
                deleted. Never returned when ``force`` is True.
        """
        need_to_free: list[MemoryObj] = []
        ret: dict[ObjectKey, L1Error] = {}
        successful_keys: list[ObjectKey] = []

        for key in keys:
            entry = self._objects.get(key, None)
            if entry is None:
                ret[key] = L1Error.KEY_NOT_EXIST
                continue

            locked = entry.read_lock.is_locked() or entry.write_lock.is_locked()
            if locked and not force:
                ret[key] = L1Error.KEY_IS_LOCKED
                continue
            if locked:
                logger.warning("L1Manager: force-deleting locked key %s", key)

            need_to_free.append(entry.memory_obj)
            del self._objects[key]
            ret[key] = L1Error.SUCCESS
            successful_keys.append(key)

        self._memory_manager.free(need_to_free)

        for listener in self._registered_listeners:
            listener.on_l1_keys_deleted_by_manager(successful_keys)
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_KEYS_EVICTED,
                metadata={"keys": successful_keys},
            )
        )
        return ret

    def touch_keys(self, keys: list[ObjectKey]):
        """Touch the given keys, marking the keys as accessed(retrieved or stored).

        Args:
            keys: The list of object keys to touch.
        """
        for listener in self._registered_listeners:
            listener.on_l1_keys_accessed(keys)

    @l1_mgr_synchronized
    def clear(self, force: bool = False) -> None:
        """Clear objects from L1 cache.

        Args:
            force: If True, clear ALL objects including locked ones.
                This may corrupt in-flight store/prefetch operations.
                If False (default), only clear unlocked objects, keeping
                write-locked and read-locked objects intact.
        """
        if force:
            logger.warning(
                "L1Manager: force-clearing all %d objects "
                "(including locked ones). This may corrupt in-flight "
                "store/prefetch operations — use with caution.",
                len(self._objects),
            )
            all_keys = list(self._objects.keys())
            all_memory_objs = [entry.memory_obj for entry in self._objects.values()]
            self._memory_manager.free(all_memory_objs)
            self._objects.clear()
            for listener in self._registered_listeners:
                listener.on_l1_keys_deleted_by_manager(all_keys)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_KEYS_EVICTED,
                    metadata={"keys": all_keys},
                )
            )
            logger.info(
                "L1Manager: cleared %d objects, 0 remaining.",
                len(all_keys),
            )
            return

        keys_to_clear: list[ObjectKey] = []
        objs_to_free: list[MemoryObj] = []
        locked_count = 0

        for key, entry in list(self._objects.items()):
            if entry.write_lock.is_locked() or entry.read_lock.is_locked():
                locked_count += 1
                continue
            keys_to_clear.append(key)
            objs_to_free.append(entry.memory_obj)

        for key in keys_to_clear:
            del self._objects[key]

        self._memory_manager.free(objs_to_free)

        if keys_to_clear:
            for listener in self._registered_listeners:
                listener.on_l1_keys_deleted_by_manager(keys_to_clear)
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_KEYS_EVICTED,
                    metadata={"keys": keys_to_clear},
                )
            )

        logger.info(
            "L1Manager: cleared %d objects, %d locked objects remaining.",
            len(keys_to_clear),
            locked_count,
        )

    @l1_mgr_synchronized
    def get_evictable_keys(
        self,
        limit: int,
        cursor: int = 0,
        scan_limit: int | None = None,
    ) -> tuple[list[ObjectKey], int]:
        """Return a bounded, rotating batch of evictable keys.

        ``cursor`` is an insertion-order offset returned by the previous call.
        The scan wraps once and inspects at most ``scan_limit`` entries, so a
        periodic backup cannot materialize the complete L1 keyspace while the
        manager lock is held. Concurrent insertions or removals may shift the
        offset, but every returned key is revalidated by ``reserve_read``.

        Args:
            limit: Maximum number of keys to return.
            cursor: Insertion-order offset at which to resume scanning.
            scan_limit: Maximum entries to inspect. Defaults to four batches.

        Returns:
            A pair of the eligible keys and the cursor for the next call.
        """
        if limit <= 0 or not self._objects:
            return [], 0

        object_count = len(self._objects)
        start = cursor % object_count
        max_scan = min(
            object_count,
            max(limit, scan_limit if scan_limit is not None else limit * 4),
        )
        ordered_keys = chain(
            islice(self._objects, start, None),
            islice(self._objects, 0, start),
        )
        keys: list[ObjectKey] = []
        scanned = 0
        for key in ordered_keys:
            scanned += 1
            if self.is_key_evictable(key):
                keys.append(key)
                if len(keys) >= limit:
                    break
            if scanned >= max_scan:
                break
        return keys, (start + scanned) % object_count

    @l1_mgr_synchronized
    def num_objects(self) -> int:
        """Return the number of objects currently tracked in L1."""
        return len(self._objects)

    def is_key_evictable(self, key: ObjectKey) -> bool:
        """Check if a key is eligible for eviction (not locked).

        This method does NOT acquire the global L1Manager lock.
        L1Manager.delete() will check again and safely reject a key
        that became locked between the check and the actual deletion.

        Args:
            key: The object key to check.

        Returns:
            True if the key exists and is not locked (neither read-locked
            nor write-locked), False otherwise.
        """
        entry = self._objects.get(key, None)
        if entry is None:
            return False
        return not entry.read_lock.is_locked() and not entry.write_lock.is_locked()

    def get_memory_usage(self) -> tuple[int, int]:
        """Get the current memory usage of L1 cache.

        Returns:
            A tuple of (used_memory_bytes, total_memory_bytes).

        Note:
            In the future, we many want to make a "callback" based mechanism
            via "L1ManagerListener" to notify the memory usage changes.
        """
        return self._memory_manager.get_memory_usage()

    def get_l1_memory_desc(self):
        """Return an L1MemoryDesc describing the underlying L1 memory buffer."""
        return self._memory_manager.get_l1_memory_desc()

    def close(self) -> None:
        """Close the L1Manager and free all resources."""
        with self._lock:
            all_memory_objs = [entry.memory_obj for entry in self._objects.values()]
            self._memory_manager.free(all_memory_objs)
            self._objects.clear()

        self._memory_manager.close()

    # Status reporting
    @l1_mgr_synchronized
    def report_status(self) -> dict:
        """Return a status dict describing L1 cache state."""
        write_locked = 0
        read_locked = 0
        temporary = 0
        for entry in self._objects.values():
            if entry.write_lock.is_locked():
                write_locked += 1
            if entry.read_lock.is_locked():
                read_locked += 1
            if entry.is_temporary:
                temporary += 1
        used, total = self._memory_manager.get_memory_usage()
        return {
            "is_healthy": self._memory_manager.memcheck(),
            "total_object_count": len(self._objects),
            "write_locked_count": write_locked,
            "read_locked_count": read_locked,
            "temporary_count": temporary,
            "memory_used_bytes": used,
            "memory_total_bytes": total,
            "memory_usage_ratio": used / total if total > 0 else 0.0,
            "write_ttl_seconds": self._write_ttl_seconds,
            "read_ttl_seconds": self._read_ttl_seconds,
        }

    # Debugging APIs
    @l1_mgr_synchronized
    def count_readable_prefix(self, keys: list[ObjectKey]) -> int:
        """Return how many leading keys are present and readable in L1.

        Presence only: no read lock is taken and no TTL is refreshed, so the
        answer can be stale by the time a caller acts on it. Used by
        lookup-only prefetches that never retrieve the objects.
        """
        count = 0
        for key in keys:
            state = self._objects.get(key, None)
            if state is None or not state.available_for_read():
                break
            count += 1
        return count

    @l1_mgr_synchronized
    def get_object_state(self, key: ObjectKey) -> L1ObjectState | None:
        """Get the internal state of the object with the given key.

        Args:
            key: The object key.

        Returns:
            The L1ObjectState if the object exists, None otherwise.
        """
        return self._objects.get(key, None)

    @l1_mgr_synchronized
    def memcheck(self) -> bool:
        """Perform memory check for L1 cache."""
        mem_check_result = self._memory_manager.memcheck()

        # Log the locked objects for debugging
        num_write_locked = 0
        num_read_locked = 0
        for key, entry in self._objects.items():
            if entry.write_lock.is_locked():
                num_write_locked += 1
            if entry.read_lock.is_locked():
                num_read_locked += 1

        logger.info(
            "L1Manager memcheck: total objects = %d, write-locked = %d, "
            "read-locked = %d",
            len(self._objects),
            num_write_locked,
            num_read_locked,
        )
        return mem_check_result
