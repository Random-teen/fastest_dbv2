"""
FastDB - Ultra-High-Performance Database Engine with Visualization

Usage:
    import fastdb

    db = fastdb.open("/tmp/mydb")
    db.put("key", "value")
    print(db.get("key"))
    db.close()

Visualization:
    fastdb.visualize.latency_histogram(db)
    fastdb.visualize.throughput_timeline(db)
    fastdb.visualize.storage_map(db)
"""

from fastdb.database import FastDB, open, DataType
from fastdb import visualize

__version__ = "1.0.0"
__all__ = ["FastDB", "open", "DataType", "visualize"]
