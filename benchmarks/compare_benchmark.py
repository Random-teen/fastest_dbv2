#!/usr/bin/env python3
"""
FastDB Benchmark Comparison Script

Benchmarks FastDB (native C extension + turbo mode), FastDB (ctypes),
SQLite (WAL mode), and Python dict.
"""

import os
import sys
import json
import time
import struct
import sqlite3
import tempfile
import shutil
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# Import native extension directly for turbo benchmarks
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python', 'fastdb'))
try:
    import _fastdb_ext
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

import fastdb
from fastdb import visualize

RECORD_COUNT = 200000
VALUE_SIZE = 100
SCREENSHOTS_DIR = os.environ.get('TWILL_SCREENSHOTS_DIR', '/tmp')


def make_key(i):
    return f"bench_key_{i:020d}"


def make_value():
    return "A" * VALUE_SIZE


def benchmark_fastdb_turbo(n):
    """Benchmark FastDB with native C extension + turbo mode."""
    if not HAS_NATIVE:
        print("  Native extension not available, skipping turbo benchmark")
        return {}

    db_path = tempfile.mktemp(prefix="fastdb_turbo_")
    results = {}

    try:
        db = _fastdb_ext.open(db_path, turbo=True)
        val = make_value()

        # Sequential insert
        t0 = time.perf_counter()
        for i in range(n):
            db[make_key(i)] = val
        elapsed = time.perf_counter() - t0
        results['seq_insert'] = n / elapsed
        print(f"  FastDB(turbo) insert:  {results['seq_insert']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Sequential read
        t0 = time.perf_counter()
        for i in range(n):
            _ = db[make_key(i)]
        elapsed = time.perf_counter() - t0
        results['seq_read'] = n / elapsed
        print(f"  FastDB(turbo) read:    {results['seq_read']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Random read
        indices = [random.randint(0, n - 1) for _ in range(n)]
        t0 = time.perf_counter()
        for idx in indices:
            _ = db[make_key(idx)]
        elapsed = time.perf_counter() - t0
        results['rand_read'] = n / elapsed
        print(f"  FastDB(turbo) rand:    {results['rand_read']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Update (overwrite with same-size value)
        new_val = "Z" * VALUE_SIZE
        t0 = time.perf_counter()
        for i in range(n):
            db[make_key(i)] = new_val
        elapsed = time.perf_counter() - t0
        results['update'] = n / elapsed
        print(f"  FastDB(turbo) update:  {results['update']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Delete
        t0 = time.perf_counter()
        for i in range(0, n, 2):
            del db[make_key(i)]
        elapsed = time.perf_counter() - t0
        results['delete'] = (n // 2) / elapsed
        print(f"  FastDB(turbo) delete:  {results['delete']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Scan
        count = [0]
        t0 = time.perf_counter()
        def _cb(k, v):
            count[0] += 1
            return True
        db.scan(_cb)
        elapsed = time.perf_counter() - t0
        results['full_scan'] = count[0] / elapsed if elapsed > 0 else 0
        print(f"  FastDB(turbo) scan:    {results['full_scan']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Batch insert (separate DB)
        db.close()
        db2_path = tempfile.mktemp(prefix="fastdb_turbo_batch_")
        db2 = _fastdb_ext.open(db2_path, turbo=True)

        pairs = [(make_key(i), val) for i in range(n)]
        batch_size = 10000
        t0 = time.perf_counter()
        for start in range(0, n, batch_size):
            batch = pairs[start:start + batch_size]
            db2.batch_put(batch)
        elapsed = time.perf_counter() - t0
        results['batch_insert'] = n / elapsed
        print(f"  FastDB(turbo) batch:   {results['batch_insert']:>12,.0f} ops/s ({elapsed:.3f}s)")

        db2.close()
        for ext in ['.fdb', '.wal', '.lock']:
            try: os.unlink(db2_path + ext)
            except: pass

    finally:
        for ext in ['.fdb', '.wal', '.lock']:
            try: os.unlink(db_path + ext)
            except: pass

    return results


def benchmark_fastdb(n):
    """Benchmark FastDB (standard ctypes wrapper)."""
    db_path = tempfile.mktemp(prefix="fastdb_bench_")
    results = {}

    try:
        db = fastdb.database.FastDB(db_path)
        db.enable_tracking()
        val = make_value()

        # Sequential insert
        t0 = time.perf_counter()
        for i in range(n):
            db.put(make_key(i), val)
        elapsed = time.perf_counter() - t0
        results['seq_insert'] = n / elapsed
        print(f"  FastDB(ctypes) insert: {results['seq_insert']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Sequential read
        t0 = time.perf_counter()
        for i in range(n):
            db.get(make_key(i))
        elapsed = time.perf_counter() - t0
        results['seq_read'] = n / elapsed
        print(f"  FastDB(ctypes) read:   {results['seq_read']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Random read
        indices = [random.randint(0, n - 1) for _ in range(n)]
        t0 = time.perf_counter()
        for idx in indices:
            db.get(make_key(idx))
        elapsed = time.perf_counter() - t0
        results['rand_read'] = n / elapsed
        print(f"  FastDB(ctypes) rand:   {results['rand_read']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Update
        new_val = "Z" * VALUE_SIZE
        t0 = time.perf_counter()
        for i in range(n):
            db.update(make_key(i), new_val)
        elapsed = time.perf_counter() - t0
        results['update'] = n / elapsed
        print(f"  FastDB(ctypes) update: {results['update']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Delete
        t0 = time.perf_counter()
        for i in range(0, n, 2):
            db.delete(make_key(i))
        elapsed = time.perf_counter() - t0
        results['delete'] = (n // 2) / elapsed
        print(f"  FastDB(ctypes) delete: {results['delete']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Scan
        count = [0]
        t0 = time.perf_counter()
        def _cb(k, v):
            count[0] += 1
            return True
        db.scan(_cb)
        elapsed = time.perf_counter() - t0
        results['full_scan'] = count[0] / elapsed if elapsed > 0 else 0
        print(f"  FastDB(ctypes) scan:   {results['full_scan']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Batch insert
        db.close()
        fastdb.database.FastDB._lib = None
        db2_path = tempfile.mktemp(prefix="fastdb_batch_")
        db2 = fastdb.database.FastDB(db2_path)
        db2.enable_tracking()

        pairs = [(make_key(i), val) for i in range(n)]
        batch_size = 10000
        t0 = time.perf_counter()
        for start in range(0, n, batch_size):
            batch = pairs[start:start + batch_size]
            db2.batch_put(batch)
        elapsed = time.perf_counter() - t0
        results['batch_insert'] = n / elapsed
        print(f"  FastDB(ctypes) batch:  {results['batch_insert']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Generate visualizations
        print("\n  Generating FastDB visualizations...")
        dash_path = os.path.join(SCREENSHOTS_DIR, "fastdb_dashboard.png")
        visualize.dashboard(db2, save=dash_path)
        print(f"    Dashboard: {dash_path}")

        lat_path = os.path.join(SCREENSHOTS_DIR, "fastdb_latency.png")
        visualize.latency_histogram(db2, save=lat_path)
        print(f"    Latency: {lat_path}")

        tp_path = os.path.join(SCREENSHOTS_DIR, "fastdb_throughput.png")
        visualize.throughput_timeline(db2, save=tp_path)
        print(f"    Throughput: {tp_path}")

        storage_path = os.path.join(SCREENSHOTS_DIR, "fastdb_storage.png")
        visualize.storage_map(db2, save=storage_path)
        print(f"    Storage: {storage_path}")

        keydist_path = os.path.join(SCREENSHOTS_DIR, "fastdb_keydist.png")
        visualize.key_distribution(db2, save=keydist_path)
        print(f"    Key dist: {keydist_path}")

        db2.close()
        fastdb.database._load_lib()
        for ext in ['.fdb', '.wal', '.lock']:
            try: os.unlink(db2_path + ext)
            except: pass

    finally:
        for ext in ['.fdb', '.wal', '.lock']:
            try: os.unlink(db_path + ext)
            except: pass

    return results


def benchmark_sqlite(n):
    """Benchmark SQLite with WAL mode for fairness."""
    db_path = tempfile.mktemp(suffix=".db", prefix="sqlite_bench_")
    results = {}

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT)")
        conn.commit()

        val = make_value()

        # Sequential insert
        t0 = time.perf_counter()
        conn.execute("BEGIN")
        for i in range(n):
            conn.execute("INSERT INTO kv VALUES (?, ?)", (make_key(i), val))
        conn.commit()
        elapsed = time.perf_counter() - t0
        results['seq_insert'] = n / elapsed
        print(f"  SQLite seq_insert:     {results['seq_insert']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Sequential read
        t0 = time.perf_counter()
        for i in range(n):
            conn.execute("SELECT value FROM kv WHERE key=?", (make_key(i),)).fetchone()
        elapsed = time.perf_counter() - t0
        results['seq_read'] = n / elapsed
        print(f"  SQLite seq_read:       {results['seq_read']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Random read
        indices = [random.randint(0, n - 1) for _ in range(n)]
        t0 = time.perf_counter()
        for idx in indices:
            conn.execute("SELECT value FROM kv WHERE key=?", (make_key(idx),)).fetchone()
        elapsed = time.perf_counter() - t0
        results['rand_read'] = n / elapsed
        print(f"  SQLite rand_read:      {results['rand_read']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Update
        new_val = "Z" * VALUE_SIZE
        t0 = time.perf_counter()
        conn.execute("BEGIN")
        for i in range(n):
            conn.execute("UPDATE kv SET value=? WHERE key=?", (new_val, make_key(i)))
        conn.commit()
        elapsed = time.perf_counter() - t0
        results['update'] = n / elapsed
        print(f"  SQLite update:         {results['update']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Delete
        t0 = time.perf_counter()
        conn.execute("BEGIN")
        for i in range(0, n, 2):
            conn.execute("DELETE FROM kv WHERE key=?", (make_key(i),))
        conn.commit()
        elapsed = time.perf_counter() - t0
        results['delete'] = (n // 2) / elapsed
        print(f"  SQLite delete:         {results['delete']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Full scan
        t0 = time.perf_counter()
        count = 0
        for row in conn.execute("SELECT * FROM kv"):
            count += 1
        elapsed = time.perf_counter() - t0
        results['full_scan'] = count / elapsed if elapsed > 0 else 0
        print(f"  SQLite full_scan:      {results['full_scan']:>12,.0f} ops/s ({elapsed:.3f}s)")

        # Batch insert
        conn.close()
        db2_path = tempfile.mktemp(suffix=".db", prefix="sqlite_batch_")
        conn2 = sqlite3.connect(db2_path)
        conn2.execute("PRAGMA journal_mode=WAL")
        conn2.execute("PRAGMA synchronous=NORMAL")
        conn2.execute("CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT)")

        pairs = [(make_key(i), val) for i in range(n)]
        t0 = time.perf_counter()
        conn2.execute("BEGIN")
        conn2.executemany("INSERT INTO kv VALUES (?, ?)", pairs)
        conn2.commit()
        elapsed = time.perf_counter() - t0
        results['batch_insert'] = n / elapsed
        print(f"  SQLite batch_insert:   {results['batch_insert']:>12,.0f} ops/s ({elapsed:.3f}s)")

        conn2.close()
        os.unlink(db2_path)

    finally:
        try: os.unlink(db_path)
        except: pass
        try: os.unlink(db_path + "-wal")
        except: pass
        try: os.unlink(db_path + "-shm")
        except: pass

    return results


def benchmark_dict(n):
    """Benchmark Python dict (in-memory baseline)."""
    results = {}
    val = make_value()

    d = {}

    # Sequential insert
    t0 = time.perf_counter()
    for i in range(n):
        d[make_key(i)] = val
    elapsed = time.perf_counter() - t0
    results['seq_insert'] = n / elapsed
    print(f"  Dict   seq_insert:     {results['seq_insert']:>12,.0f} ops/s ({elapsed:.3f}s)")

    # Sequential read
    t0 = time.perf_counter()
    for i in range(n):
        _ = d[make_key(i)]
    elapsed = time.perf_counter() - t0
    results['seq_read'] = n / elapsed
    print(f"  Dict   seq_read:       {results['seq_read']:>12,.0f} ops/s ({elapsed:.3f}s)")

    # Random read
    indices = [random.randint(0, n - 1) for _ in range(n)]
    t0 = time.perf_counter()
    for idx in indices:
        _ = d[make_key(idx)]
    elapsed = time.perf_counter() - t0
    results['rand_read'] = n / elapsed
    print(f"  Dict   rand_read:      {results['rand_read']:>12,.0f} ops/s ({elapsed:.3f}s)")

    # Update
    new_val = "Z" * VALUE_SIZE
    t0 = time.perf_counter()
    for i in range(n):
        d[make_key(i)] = new_val
    elapsed = time.perf_counter() - t0
    results['update'] = n / elapsed
    print(f"  Dict   update:         {results['update']:>12,.0f} ops/s ({elapsed:.3f}s)")

    # Delete
    t0 = time.perf_counter()
    for i in range(0, n, 2):
        del d[make_key(i)]
    elapsed = time.perf_counter() - t0
    results['delete'] = (n // 2) / elapsed
    print(f"  Dict   delete:         {results['delete']:>12,.0f} ops/s ({elapsed:.3f}s)")

    # Scan
    t0 = time.perf_counter()
    count = 0
    for k, v in d.items():
        count += 1
    elapsed = time.perf_counter() - t0
    results['full_scan'] = count / elapsed if elapsed > 0 else 0
    print(f"  Dict   full_scan:      {results['full_scan']:>12,.0f} ops/s ({elapsed:.3f}s)")

    results['batch_insert'] = results['seq_insert']

    return results


def main():
    n = RECORD_COUNT
    print()
    print("=" * 76)
    print("  FastDB Comprehensive Benchmark Comparison")
    print(f"  Records: {n:,}  |  Value size: {VALUE_SIZE} bytes")
    print(f"  Native C extension: {'YES' if HAS_NATIVE else 'NO'}")
    print("=" * 76)

    print(f"\n── FastDB (turbo — C extension, no WAL/CRC/lock) {'─' * 28}")
    turbo_results = benchmark_fastdb_turbo(n)

    print(f"\n── FastDB (ctypes — standard mode) {'─' * 40}")
    fastdb_results = benchmark_fastdb(n)

    print(f"\n── SQLite (WAL mode) {'─' * 54}")
    sqlite_results = benchmark_sqlite(n)

    print(f"\n── Python dict (in-memory, no persistence) {'─' * 32}")
    dict_results = benchmark_dict(n)

    # Generate comparison chart
    print(f"\n── Generating comparison chart {'─' * 44}")

    comparison = {
        'FastDB (turbo)': turbo_results,
        'FastDB (std)': fastdb_results,
        'SQLite (WAL)': sqlite_results,
        'Python dict*': dict_results,
    }

    chart_path = os.path.join(SCREENSHOTS_DIR, "fastdb_comparison.png")
    visualize.benchmark_comparison(comparison, metric='ops_per_sec', save=chart_path)
    print(f"  Comparison chart: {chart_path}")

    # Save raw results
    results_path = os.path.join(SCREENSHOTS_DIR, "benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"  Raw results: {results_path}")

    # Print summary table
    ops = ['seq_insert', 'seq_read', 'rand_read', 'update', 'delete', 'full_scan', 'batch_insert']

    print(f"\n{'=' * 76}")
    print("  SUMMARY (ops/sec — higher is better)")
    print(f"{'=' * 76}")
    print(f"  {'Op':<14} {'Turbo':>10} {'Std':>10} {'SQLite':>10} {'Dict*':>10}  {'Turbo/Dict':>10}")
    print(f"  {'─' * 14} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}  {'─' * 10}")

    for op in ops:
        t = turbo_results.get(op, 0)
        f = fastdb_results.get(op, 0)
        s = sqlite_results.get(op, 0)
        d = dict_results.get(op, 0)
        pct = (t / d * 100) if d > 0 else 0
        print(f"  {op:<14} {t:>10,.0f} {f:>10,.0f} {s:>10,.0f} {d:>10,.0f}  {pct:>8.0f}%")

    # Overall averages
    turbo_avg = sum(turbo_results.get(op, 0) / dict_results.get(op, 1) for op in ops[:5]) / 5 * 100
    print(f"\n  FastDB(turbo) vs Dict average (core ops): {turbo_avg:.0f}%")
    print(f"\n  * Dict is in-memory only (no persistence)")
    print(f"  FastDB turbo: mmap-backed with lazy durability")
    print(f"  FastDB std: full ACID guarantees with WAL + mmap\n")


if __name__ == '__main__':
    main()
