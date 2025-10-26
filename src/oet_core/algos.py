"""Algorithm implementations for the oet_core package."""

from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

from .utils import log


_VERBOSE_LOGGING = False


def set_verbose_logging(enabled: bool) -> None:
    """Enable or disable verbose logging for this module."""
    global _VERBOSE_LOGGING
    _VERBOSE_LOGGING = bool(enabled)


def _iter_pairs_left(
    data: List[Union[Tuple[Any, Any], List[Any], Any]],
    index: int,
    x_target: Union[int, float, str],
) -> Iterator[int]:
    """Yield indices to the left that share the same x coordinate."""
    current = index - 1
    while current >= 0:
        item = data[current]
        other_x = item[0] if isinstance(item, (list, tuple)) else item
        if other_x != x_target:
            break
        yield current
        current -= 1


def _iter_pairs_right(
    data: List[Union[Tuple[Any, Any], List[Any], Any]],
    index: int,
    x_target: Union[int, float, str],
) -> Iterator[int]:
    """Yield indices to the right that share the same x coordinate."""
    current = index + 1
    size = len(data)
    while current < size:
        item = data[current]
        other_x = item[0] if isinstance(item, (list, tuple)) else item
        if other_x != x_target:
            break
        yield current
        current += 1


def binary_search(
    pairs: List[Union[Tuple[Any, Any], List[Any], int, float, str]],
    x_target: Union[int, float, str],
    y_target: Optional[Union[int, float, str]] = None,
) -> Optional[int]:
    """Perform binary search on a sorted list.

    Supports two data layouts:
    * 1D lists of scalars: ``[1, 2, 3, ...]``
    * 2D lists/tuples storing pairs: ``[(x, y), ...]``

    Parameters
    ----------
    pairs:
        Sorted list to search. For pair inputs the first element is treated as
        the x coordinate.
    x_target:
        Value to search for in the x coordinate.
    y_target:
        Optional secondary value. When supplied, the function returns the index
        that matches both ``x_target`` and ``y_target``. When omitted, the
        left-most index with ``x_target`` is returned.
    """
    if _VERBOSE_LOGGING:
        log(
            f"algos.binary_search called with x_target={x_target}, y_target={y_target}",
            level="info",
        )

    if not isinstance(pairs, list):
        raise TypeError("pairs must be a list")

    size = len(pairs)
    if size == 0:
        return None

    lo, hi = 0, size - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        item = pairs[mid]

        if isinstance(item, (list, tuple)) and item:
            x_value = item[0]
            y_value = item[1] if len(item) >= 2 else None
        else:
            x_value = item
            y_value = None

        if x_value == x_target:
            if y_target is None:
                candidate = mid
                for idx in _iter_pairs_left(pairs, mid, x_target):
                    candidate = idx
                return candidate

            if y_value == y_target:
                return mid

            for idx in _iter_pairs_left(pairs, mid, x_target):
                other = pairs[idx]
                other_y = other[1] if isinstance(other, (list, tuple)) and len(other) >= 2 else None
                if other_y == y_target:
                    return idx

            for idx in _iter_pairs_right(pairs, mid, x_target):
                other = pairs[idx]
                other_y = other[1] if isinstance(other, (list, tuple)) and len(other) >= 2 else None
                if other_y == y_target:
                    return idx

            return None

        if x_value < x_target:
            lo = mid + 1
        else:
            hi = mid - 1

    return None


class HashMap:
    """Simple hash map implemented with separate chaining."""

    def __init__(self, initial_capacity: int = 8) -> None:
        if _VERBOSE_LOGGING:
            log(f"HashMap.__init__ called with initial_capacity={initial_capacity}", level="info")

        if not isinstance(initial_capacity, int) or initial_capacity <= 0:
            raise ValueError("initial_capacity must be a positive int")

        self._capacity = int(initial_capacity)
        self._buckets: List[List[Tuple[Any, Any]]] = [[] for _ in range(self._capacity)]
        self._size = 0

    def _bucket_index(self, key: Any) -> int:
        if _VERBOSE_LOGGING:
            log(f"HashMap._bucket_index called with key={key}", level="info")
        hashed = hash(key)
        return (hashed & 0x7FFFFFFF) % self._capacity

    def _needs_resize(self) -> bool:
        return self._size / self._capacity > 0.75

    def _resize(self, new_capacity: int) -> None:
        if _VERBOSE_LOGGING:
            log(f"HashMap._resize called with new_capacity={new_capacity}", level="info")

        old_buckets = self._buckets
        self._capacity = max(3, int(new_capacity))
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0

        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)

    def put(self, key: Any, value: Any) -> None:
        """Insert or replace a key/value pair."""
        if _VERBOSE_LOGGING:
            log(f"HashMap.put called with key={key}, value={value}", level="info")

        index = self._bucket_index(key)
        bucket = self._buckets[index]

        for pos, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                bucket[pos] = (key, value)
                return

        bucket.append((key, value))
        self._size += 1

        if self._needs_resize():
            self._resize(self._capacity * 2)

    def get(self, key: Any, default: Any = None) -> Any:
        """Retrieve the value for *key* or return *default*."""
        if _VERBOSE_LOGGING:
            log(f"HashMap.get called with key={key}, default={default}", level="info")

        index = self._bucket_index(key)
        for stored_key, stored_value in self._buckets[index]:
            if stored_key == key:
                return stored_value
        return default

    def delete(self, key: Any) -> bool:
        """Remove *key* from the map. Returns ``True`` when removed."""
        if _VERBOSE_LOGGING:
            log(f"HashMap.delete called with key={key}", level="info")

        index = self._bucket_index(key)
        bucket = self._buckets[index]

        for pos, (stored_key, _) in enumerate(bucket):
            if stored_key == key:
                del bucket[pos]
                self._size -= 1
                return True
        return False

    def contains(self, key: Any) -> bool:
        """Return ``True`` when *key* exists in the map."""
        if _VERBOSE_LOGGING:
            log(f"HashMap.contains called with key={key}", level="info")

        index = self._bucket_index(key)
        return any(stored_key == key for stored_key, _ in self._buckets[index])

    def keys(self) -> Iterable[Any]:
        """Yield keys currently stored in the map."""
        if _VERBOSE_LOGGING:
            log("HashMap.keys called", level="info")
        for bucket in self._buckets:
            for stored_key, _ in bucket:
                yield stored_key

    def values(self) -> Iterable[Any]:
        """Yield values currently stored in the map."""
        if _VERBOSE_LOGGING:
            log("HashMap.values called", level="info")
        for bucket in self._buckets:
            for _, stored_value in bucket:
                yield stored_value

    def items(self) -> Iterable[Tuple[Any, Any]]:
        """Yield ``(key, value)`` pairs stored in the map."""
        if _VERBOSE_LOGGING:
            log("HashMap.items called", level="info")
        for bucket in self._buckets:
            for stored_key, stored_value in bucket:
                yield stored_key, stored_value

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"HashMap(size={self._size}, capacity={self._capacity})"

