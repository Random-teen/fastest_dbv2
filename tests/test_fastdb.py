#!/usr/bin/env python3
"""FastDB test suite."""

import os
import sys
import json
import tempfile
import struct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import fastdb
from fastdb.database import NotFoundError

DB_PATH = None
db = None


def setup():
    global DB_PATH, db
    DB_PATH = tempfile.mktemp(prefix="fastdb_test_")
    db = fastdb.open(DB_PATH)


def teardown():
    global db, DB_PATH
    if db:
        db.close()
        db = None
    if DB_PATH:
        for ext in ['.fdb', '.wal', '.lock']:
            try:
                os.unlink(DB_PATH + ext)
            except FileNotFoundError:
                pass


def test_put_get_string():
    setup()
    try:
        db.put("hello", "world")
        assert db.get("hello") == "world", "String put/get failed"
        print("  PASS: put/get string")
    finally:
        teardown()


def test_put_get_int():
    setup()
    try:
        db.put("num", 42)
        result = db.get("num")
        assert result == 42, f"Int put/get failed: got {result}"
        print("  PASS: put/get int")
    finally:
        teardown()


def test_put_get_float():
    setup()
    try:
        db.put("pi", 3.14159)
        result = db.get("pi")
        assert abs(result - 3.14159) < 1e-10, f"Float put/get failed: got {result}"
        print("  PASS: put/get float")
    finally:
        teardown()


def test_put_get_bytes():
    setup()
    try:
        data = b"\x00\x01\x02\xff\xfe"
        db.put("binary", data)
        result = db.get("binary")
        assert result == data, f"Bytes put/get failed"
        print("  PASS: put/get bytes")
    finally:
        teardown()


def test_put_get_json():
    setup()
    try:
        obj = {"name": "FastDB", "version": 1, "fast": True}
        db.put("config", obj)
        result = db.get("config")
        assert result == obj, f"JSON put/get failed: got {result}"
        print("  PASS: put/get JSON object")
    finally:
        teardown()


def test_put_get_list():
    setup()
    try:
        arr = [1, 2, 3, "four", 5.0]
        db.put("list", arr)
        result = db.get("list")
        assert result == arr, f"List put/get failed"
        print("  PASS: put/get list")
    finally:
        teardown()


def test_get_missing():
    setup()
    try:
        result = db.get("nonexistent")
        assert result is None, "Missing key should return None"
        result = db.get("nonexistent", "default")
        assert result == "default", "Missing key should return default"
        print("  PASS: get missing key")
    finally:
        teardown()


def test_delete():
    setup()
    try:
        db.put("to_delete", "bye")
        assert db.get("to_delete") == "bye"
        db.delete("to_delete")
        assert db.get("to_delete") is None, "Deleted key should be None"
        print("  PASS: delete")
    finally:
        teardown()


def test_update():
    setup()
    try:
        db.put("key", "v1")
        assert db.get("key") == "v1"
        db.update("key", "v2")
        assert db.get("key") == "v2", "Update failed"
        print("  PASS: update")
    finally:
        teardown()


def test_update_missing():
    setup()
    try:
        try:
            db.update("missing", "value")
            assert False, "Should have raised NotFoundError"
        except NotFoundError:
            pass
        print("  PASS: update missing raises error")
    finally:
        teardown()


def test_exists():
    setup()
    try:
        db.put("exists_key", "yes")
        assert db.exists("exists_key") is True
        assert db.exists("nope") is False
        assert "exists_key" in db
        print("  PASS: exists")
    finally:
        teardown()


def test_overwrite():
    setup()
    try:
        db.put("ow", "first")
        db.put("ow", "second")
        assert db.get("ow") == "second", "Overwrite failed"
        print("  PASS: overwrite (put same key)")
    finally:
        teardown()


def test_scan():
    setup()
    try:
        for i in range(100):
            db.put(f"scan_{i:04d}", f"value_{i}")

        items = db.items()
        assert len(items) == 100, f"Scan returned {len(items)} items, expected 100"

        keys = db.keys()
        assert len(keys) == 100
        print("  PASS: scan (100 records)")
    finally:
        teardown()


def test_batch_put():
    setup()
    try:
        pairs = [(f"batch_{i:06d}", f"val_{i}") for i in range(1000)]
        db.batch_put(pairs)
        assert len(db) == 1000, f"Batch put: got {len(db)} records"

        # Verify a few
        assert db.get("batch_000000") == "val_0"
        assert db.get("batch_000999") == "val_999"
        print("  PASS: batch_put (1000 records)")
    finally:
        teardown()


def test_dict_interface():
    setup()
    try:
        db["key1"] = "val1"
        assert db["key1"] == "val1"
        del db["key1"]
        try:
            _ = db["key1"]
            assert False, "Should raise KeyError"
        except KeyError:
            pass
        print("  PASS: dict-style interface")
    finally:
        teardown()


def test_persistence():
    """Test that data persists across open/close cycles."""
    path = tempfile.mktemp(prefix="fastdb_persist_")
    try:
        db1 = fastdb.open(path)
        db1.put("persist", "forever")
        db1.sync()
        db1.close()

        db2 = fastdb.open(path)
        result = db2.get("persist")
        assert result == "forever", f"Persistence failed: got {result}"
        db2.close()
        print("  PASS: persistence across open/close")
    finally:
        for ext in ['.fdb', '.wal', '.lock']:
            try: os.unlink(path + ext)
            except: pass


def test_large_value():
    setup()
    try:
        large = "X" * (1024 * 1024)  # 1 MB
        db.put("large", large)
        result = db.get("large")
        assert result == large, "Large value failed"
        print("  PASS: large value (1 MB)")
    finally:
        teardown()


def test_context_manager():
    path = tempfile.mktemp(prefix="fastdb_ctx_")
    try:
        with fastdb.open(path) as db_ctx:
            db_ctx.put("ctx", "test")
            assert db_ctx.get("ctx") == "test"
        print("  PASS: context manager")
    finally:
        for ext in ['.fdb', '.wal', '.lock']:
            try: os.unlink(path + ext)
            except: pass


def test_stats():
    setup()
    try:
        for i in range(50):
            db.put(f"stat_{i}", f"value_{i}")

        stats = db.stats()
        assert stats['record_count'] == 50, f"Stats count wrong: {stats['record_count']}"
        assert stats['writes'] >= 50
        assert stats['data_size'] > 0
        print("  PASS: stats")
    finally:
        teardown()


def test_mixed_types():
    setup()
    try:
        db.put("str", "hello")
        db.put("int", 42)
        db.put("float", 3.14)
        db.put("bytes", b"\x00\xff")
        db.put("json", {"a": 1})
        db.put("list", [1, 2, 3])

        assert db.get("str") == "hello"
        assert db.get("int") == 42
        assert abs(db.get("float") - 3.14) < 0.01
        assert db.get("bytes") == b"\x00\xff"
        assert db.get("json") == {"a": 1}
        assert db.get("list") == [1, 2, 3]
        print("  PASS: mixed types")
    finally:
        teardown()


def main():
    print("\n  FastDB Test Suite")
    print("  " + "=" * 40)

    tests = [
        test_put_get_string,
        test_put_get_int,
        test_put_get_float,
        test_put_get_bytes,
        test_put_get_json,
        test_put_get_list,
        test_get_missing,
        test_delete,
        test_update,
        test_update_missing,
        test_exists,
        test_overwrite,
        test_scan,
        test_batch_put,
        test_dict_interface,
        test_persistence,
        test_large_value,
        test_context_manager,
        test_stats,
        test_mixed_types,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")

    print(f"\n  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print()

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
