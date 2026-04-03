"""
FastDB - Ultra-High-Performance Database Engine with Visualization

Usage:
    import fastdb

    # Standard mode (with WAL, CRC, locking — full durability)
    db = fastdb.open("/tmp/mydb")
    db.put("key", "value")
    print(db.get("key"))
    db.close()

    # Turbo mode (dict-speed performance, lazy durability)
    db = fastdb.open("/tmp/mydb", turbo=True)
    db["key"] = "value"
    print(db["key"])
    db.close()

Visualization:
    fastdb.visualize.latency_histogram(db)
    fastdb.visualize.throughput_timeline(db)
    fastdb.visualize.storage_map(db)
"""

try:
    from fastdb._fastdb_ext import FastDB as _NativeFastDB, open as _native_open
    from fastdb._fastdb_ext import FastDBError, NotFoundError
    from fastdb._fastdb_ext import (
        FLAG_DEFAULT, FLAG_NO_WAL, FLAG_NO_CRC,
        FLAG_NO_LOCK, FLAG_NO_STATS, FLAG_TURBO,
    )
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

from fastdb.database import FastDB as _CtypesFastDB
from fastdb.database import open as _ctypes_open
from fastdb.database import DataType
from fastdb import visualize


def open(path, turbo=False):
    """Open a FastDB database. Uses native C extension when available."""
    if HAS_NATIVE:
        return _native_open(path, turbo=turbo)
    return _ctypes_open(path)


if HAS_NATIVE:
    FastDB = _NativeFastDB
else:
    FastDB = _CtypesFastDB


__version__ = "1.0.0"
__all__ = [
    "FastDB", "open", "DataType", "visualize",
    "FLAG_DEFAULT", "FLAG_NO_WAL", "FLAG_NO_CRC",
    "FLAG_NO_LOCK", "FLAG_NO_STATS", "FLAG_TURBO",
]
