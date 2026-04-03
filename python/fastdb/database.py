"""
FastDB Python wrapper — ctypes bindings to the C engine.
"""

import ctypes
import ctypes.util
import os
import time
import json
import struct
import threading
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional, Union, Dict, List, Callable, Iterator, Tuple
from collections import defaultdict


class DataType(IntEnum):
    RAW = 0
    INT64 = 1
    UINT64 = 2
    DOUBLE = 3
    STRING = 4
    BLOB = 5
    JSON = 6
    ARRAY = 7
    MAP = 8


class FastDBError(Exception):
    """Base exception for FastDB errors."""
    pass


class NotFoundError(FastDBError):
    pass


class CorruptError(FastDBError):
    pass


# Error code mapping
_ERROR_MAP = {
    -1: (FastDBError, "I/O error"),
    -2: (CorruptError, "Database corrupt"),
    -3: (FastDBError, "Database full"),
    -4: (NotFoundError, "Key not found"),
    -5: (FastDBError, "Key already exists"),
    -6: (FastDBError, "Invalid argument"),
    -7: (FastDBError, "Out of memory"),
    -8: (FastDBError, "Database locked"),
    -9: (FastDBError, "WAL error"),
    -10: (FastDBError, "Memory map error"),
}


def _check(rc: int):
    if rc != 0:
        exc_class, msg = _ERROR_MAP.get(rc, (FastDBError, f"Unknown error {rc}"))
        raise exc_class(msg)


# ============================================================
# ctypes structures
# ============================================================

class _Slice(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("len", ctypes.c_uint32),
    ]


class _Value(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("data", _Slice),
        ("timestamp", ctypes.c_uint64),
    ]


class _Stats(ctypes.Structure):
    _fields_ = [
        ("record_count", ctypes.c_uint64),
        ("data_size", ctypes.c_uint64),
        ("index_buckets", ctypes.c_uint64),
        ("wal_lsn", ctypes.c_uint64),
        ("reads", ctypes.c_uint64),
        ("writes", ctypes.c_uint64),
        ("deletes", ctypes.c_uint64),
        ("scans", ctypes.c_uint64),
        ("avg_read_ns", ctypes.c_double),
        ("avg_write_ns", ctypes.c_double),
        ("load_factor", ctypes.c_double),
    ]


# Scan callback type
_SCAN_CB = ctypes.CFUNCTYPE(
    ctypes.c_bool,
    ctypes.POINTER(_Slice),   # key
    ctypes.POINTER(_Value),   # value
    ctypes.c_void_p,          # user_data
)


def _find_lib():
    """Find the FastDB shared library."""
    search_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'libfastdb.so'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'libfastdb.so'),
        '/usr/local/lib/libfastdb.so',
        './build/libfastdb.so',
        './libfastdb.so',
    ]
    for p in search_paths:
        p = os.path.abspath(p)
        if os.path.exists(p):
            return p
    return None


def _load_lib():
    path = _find_lib()
    if not path:
        raise FastDBError(
            "libfastdb.so not found. Build with: make -C <project_root>"
        )
    lib = ctypes.CDLL(path)

    # fastdb_open
    lib.fastdb_open.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p
    ]
    lib.fastdb_open.restype = ctypes.c_int

    # fastdb_close
    lib.fastdb_close.argtypes = [ctypes.c_void_p]
    lib.fastdb_close.restype = ctypes.c_int

    # fastdb_destroy
    lib.fastdb_destroy.argtypes = [ctypes.c_char_p]
    lib.fastdb_destroy.restype = ctypes.c_int

    # fastdb_put
    lib.fastdb_put.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Slice),
        ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_int,
    ]
    lib.fastdb_put.restype = ctypes.c_int

    # fastdb_get
    lib.fastdb_get.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Slice),
        ctypes.POINTER(_Value),
    ]
    lib.fastdb_get.restype = ctypes.c_int

    # fastdb_delete
    lib.fastdb_delete.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Slice),
    ]
    lib.fastdb_delete.restype = ctypes.c_int

    # fastdb_update
    lib.fastdb_update.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Slice),
        ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_int,
    ]
    lib.fastdb_update.restype = ctypes.c_int

    # fastdb_exists
    lib.fastdb_exists.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Slice),
        ctypes.POINTER(ctypes.c_bool),
    ]
    lib.fastdb_exists.restype = ctypes.c_int

    # fastdb_scan
    lib.fastdb_scan.argtypes = [
        ctypes.c_void_p, _SCAN_CB, ctypes.c_void_p,
    ]
    lib.fastdb_scan.restype = ctypes.c_int

    # fastdb_sync
    lib.fastdb_sync.argtypes = [ctypes.c_void_p]
    lib.fastdb_sync.restype = ctypes.c_int

    # fastdb_checkpoint
    lib.fastdb_checkpoint.argtypes = [ctypes.c_void_p]
    lib.fastdb_checkpoint.restype = ctypes.c_int

    # fastdb_stats
    lib.fastdb_stats.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_Stats),
    ]
    lib.fastdb_stats.restype = ctypes.c_int

    # fastdb_batch_put
    lib.fastdb_batch_put.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Slice),
        ctypes.POINTER(_Slice),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint32,
    ]
    lib.fastdb_batch_put.restype = ctypes.c_int

    return lib


class FastDB:
    """High-level Python interface to FastDB."""

    def __init__(self, path: str):
        self._lib = _load_lib()
        self._handle = ctypes.c_void_p()
        self._path = path

        rc = self._lib.fastdb_open(
            ctypes.byref(self._handle),
            path.encode('utf-8')
        )
        _check(rc)

        # Performance tracking for visualization
        self._op_log: List[Dict] = []
        self._track = False
        self._lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def path(self) -> str:
        return self._path

    def enable_tracking(self):
        """Enable operation latency tracking for visualization."""
        self._track = True

    def disable_tracking(self):
        """Disable operation latency tracking."""
        self._track = False

    def get_op_log(self) -> List[Dict]:
        """Return the operation log for visualization."""
        return list(self._op_log)

    def clear_op_log(self):
        self._op_log.clear()

    def _log_op(self, op: str, latency_ns: float, key_size: int = 0,
                val_size: int = 0):
        if self._track:
            with self._lock:
                self._op_log.append({
                    'op': op,
                    'latency_ns': latency_ns,
                    'key_size': key_size,
                    'val_size': val_size,
                    'timestamp': time.time(),
                })

    def _make_slice(self, data: Union[str, bytes]) -> Tuple[_Slice, bytes]:
        """Create a Slice from str or bytes. Returns (slice, buf) — keep buf alive."""
        if isinstance(data, str):
            buf = data.encode('utf-8')
        else:
            buf = bytes(data)
        s = _Slice()
        s.data = ctypes.cast(ctypes.c_char_p(buf), ctypes.c_void_p)
        s.len = len(buf)
        return s, buf

    def _encode_value(self, value: Any) -> Tuple[bytes, DataType]:
        """Encode a Python value to bytes + type tag."""
        if isinstance(value, bytes):
            return value, DataType.BLOB
        elif isinstance(value, str):
            return value.encode('utf-8'), DataType.STRING
        elif isinstance(value, int):
            return struct.pack('<q', value), DataType.INT64
        elif isinstance(value, float):
            return struct.pack('<d', value), DataType.DOUBLE
        elif isinstance(value, (dict, list)):
            return json.dumps(value).encode('utf-8'), DataType.JSON
        else:
            return bytes(value), DataType.RAW

    def _decode_value(self, val: _Value) -> Any:
        """Decode a Value to a Python object."""
        data = ctypes.string_at(val.data.data, val.data.len)
        t = val.type

        if t == DataType.STRING:
            return data.decode('utf-8')
        elif t == DataType.INT64:
            return struct.unpack('<q', data)[0]
        elif t == DataType.UINT64:
            return struct.unpack('<Q', data)[0]
        elif t == DataType.DOUBLE:
            return struct.unpack('<d', data)[0]
        elif t == DataType.JSON:
            return json.loads(data.decode('utf-8'))
        elif t == DataType.BLOB:
            return data
        else:
            return data

    # ======================== CRUD ========================

    def put(self, key: Union[str, bytes], value: Any,
            dtype: Optional[DataType] = None) -> None:
        """Insert or overwrite a key-value pair."""
        key_slice, key_buf = self._make_slice(key)
        val_bytes, auto_type = self._encode_value(value)
        if dtype is not None:
            auto_type = dtype

        t0 = time.perf_counter_ns()
        rc = self._lib.fastdb_put(
            self._handle,
            ctypes.byref(key_slice),
            val_bytes, len(val_bytes),
            int(auto_type),
        )
        elapsed = time.perf_counter_ns() - t0
        _check(rc)
        self._log_op('put', elapsed, len(key_buf), len(val_bytes))

    def get(self, key: Union[str, bytes], default: Any = None) -> Any:
        """Retrieve a value by key. Returns default if not found."""
        key_slice, key_buf = self._make_slice(key)
        val = _Value()

        t0 = time.perf_counter_ns()
        rc = self._lib.fastdb_get(
            self._handle,
            ctypes.byref(key_slice),
            ctypes.byref(val),
        )
        elapsed = time.perf_counter_ns() - t0

        if rc == -4:  # NOT_FOUND
            self._log_op('get_miss', elapsed, len(key_buf))
            return default
        _check(rc)
        self._log_op('get', elapsed, len(key_buf), val.data.len)
        return self._decode_value(val)

    def delete(self, key: Union[str, bytes]) -> None:
        """Delete a key."""
        key_slice, key_buf = self._make_slice(key)

        t0 = time.perf_counter_ns()
        rc = self._lib.fastdb_delete(
            self._handle,
            ctypes.byref(key_slice),
        )
        elapsed = time.perf_counter_ns() - t0
        _check(rc)
        self._log_op('delete', elapsed, len(key_buf))

    def update(self, key: Union[str, bytes], value: Any,
               dtype: Optional[DataType] = None) -> None:
        """Update an existing key. Raises NotFoundError if missing."""
        key_slice, key_buf = self._make_slice(key)
        val_bytes, auto_type = self._encode_value(value)
        if dtype is not None:
            auto_type = dtype

        t0 = time.perf_counter_ns()
        rc = self._lib.fastdb_update(
            self._handle,
            ctypes.byref(key_slice),
            val_bytes, len(val_bytes),
            int(auto_type),
        )
        elapsed = time.perf_counter_ns() - t0
        _check(rc)
        self._log_op('update', elapsed, len(key_buf), len(val_bytes))

    def exists(self, key: Union[str, bytes]) -> bool:
        """Check if a key exists."""
        key_slice, _ = self._make_slice(key)
        result = ctypes.c_bool(False)
        rc = self._lib.fastdb_exists(
            self._handle,
            ctypes.byref(key_slice),
            ctypes.byref(result),
        )
        _check(rc)
        return result.value

    def __contains__(self, key):
        return self.exists(key)

    def __getitem__(self, key):
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __setitem__(self, key, value):
        self.put(key, value)

    def __delitem__(self, key):
        self.delete(key)

    # ======================== Scan ========================

    def scan(self, callback: Callable[[bytes, Any], bool]) -> None:
        """Full database scan. Callback receives (key_bytes, value) and
        should return True to continue, False to stop."""
        results = []

        @_SCAN_CB
        def _cb(key_ptr, val_ptr, _ud):
            key_data = ctypes.string_at(key_ptr.contents.data,
                                        key_ptr.contents.len)
            val = self._decode_value(val_ptr.contents)
            return callback(key_data, val)

        t0 = time.perf_counter_ns()
        rc = self._lib.fastdb_scan(self._handle, _cb, None)
        elapsed = time.perf_counter_ns() - t0
        self._log_op('scan', elapsed)
        _check(rc)

    def items(self) -> List[Tuple[bytes, Any]]:
        """Return all key-value pairs."""
        results = []
        def _collect(k, v):
            results.append((k, v))
            return True
        self.scan(_collect)
        return results

    def keys(self) -> List[bytes]:
        """Return all keys."""
        return [k for k, _ in self.items()]

    def values(self) -> List[Any]:
        """Return all values."""
        return [v for _, v in self.items()]

    def __len__(self):
        return self.stats()['record_count']

    # ======================== Batch ========================

    def batch_put(self, pairs: List[Tuple[Union[str, bytes], Any]]) -> None:
        """Bulk insert key-value pairs. Much faster than individual puts."""
        n = len(pairs)
        if n == 0:
            return

        keys_arr = (_Slice * n)()
        vals_arr = (_Slice * n)()
        types_arr = (ctypes.c_int * n)()

        # Keep references alive
        key_bufs = []
        val_bufs = []

        for i, (k, v) in enumerate(pairs):
            ks, kb = self._make_slice(k)
            keys_arr[i] = ks
            key_bufs.append(kb)

            vb, vt = self._encode_value(v)
            vs = _Slice()
            vs.data = ctypes.cast(ctypes.c_char_p(vb), ctypes.c_void_p)
            vs.len = len(vb)
            vals_arr[i] = vs
            val_bufs.append(vb)
            types_arr[i] = int(vt)

        t0 = time.perf_counter_ns()
        rc = self._lib.fastdb_batch_put(
            self._handle,
            keys_arr, vals_arr, types_arr, n
        )
        elapsed = time.perf_counter_ns() - t0
        _check(rc)
        self._log_op('batch_put', elapsed, 0, n)

    # ======================== Durability ========================

    def sync(self) -> None:
        """Force sync data and WAL to disk."""
        _check(self._lib.fastdb_sync(self._handle))

    def checkpoint(self) -> None:
        """Checkpoint: sync and truncate WAL."""
        _check(self._lib.fastdb_checkpoint(self._handle))

    # ======================== Stats ========================

    def stats(self) -> Dict[str, Any]:
        """Return database statistics."""
        s = _Stats()
        _check(self._lib.fastdb_stats(self._handle, ctypes.byref(s)))
        return {
            'record_count': s.record_count,
            'data_size': s.data_size,
            'data_size_mb': round(s.data_size / (1024 * 1024), 2),
            'index_buckets': s.index_buckets,
            'wal_lsn': s.wal_lsn,
            'reads': s.reads,
            'writes': s.writes,
            'deletes': s.deletes,
            'scans': s.scans,
            'load_factor': round(s.load_factor, 4),
        }

    # ======================== Close ========================

    def close(self) -> None:
        """Close the database."""
        if self._handle and self._handle.value:
            self._lib.fastdb_close(self._handle)
            self._handle = ctypes.c_void_p()


def open(path: str) -> FastDB:
    """Open or create a FastDB database at the given path."""
    return FastDB(path)
