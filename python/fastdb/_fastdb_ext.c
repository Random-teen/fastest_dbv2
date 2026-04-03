/*
 * FastDB Native CPython Extension Module
 *
 * Replaces the ctypes wrapper with zero-overhead direct C calls.
 * Implements the Python mapping protocol for dict-like performance.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "fastdb.h"
#include <string.h>

/* ============================================================
 * FastDB Python object
 * ============================================================ */

typedef struct {
    PyObject_HEAD
    fastdb_t *db;
    int closed;
} FastDBObject;

static PyTypeObject FastDBType;

/* Forward declarations */
static PyObject *FastDB_put(FastDBObject *self, PyObject *args, PyObject *kwds);
static PyObject *FastDB_get(FastDBObject *self, PyObject *args, PyObject *kwds);

/* ============================================================
 * Helpers: encode/decode Python values to/from raw bytes
 * ============================================================ */

/* Encode a Python value to bytes + type. Returns 0 on success, -1 on error.
 * Caller must free *out_data if *out_needs_free is set. */
static int encode_value(PyObject *value,
                        const char **out_data, uint32_t *out_len,
                        fastdb_type_t *out_type, int *out_needs_free) {
    *out_needs_free = 0;

    if (PyUnicode_Check(value)) {
        Py_ssize_t len;
        const char *utf8 = PyUnicode_AsUTF8AndSize(value, &len);
        if (!utf8) return -1;
        *out_data = utf8;
        *out_len = (uint32_t)len;
        *out_type = FASTDB_TYPE_STRING;
        return 0;
    }

    if (PyBytes_Check(value)) {
        *out_data = PyBytes_AS_STRING(value);
        *out_len = (uint32_t)PyBytes_GET_SIZE(value);
        *out_type = FASTDB_TYPE_BLOB;
        return 0;
    }

    if (PyLong_Check(value)) {
        int64_t v = PyLong_AsLongLong(value);
        if (v == -1 && PyErr_Occurred()) return -1;
        /* We need to allocate a buffer for the int64 */
        char *buf = PyMem_Malloc(8);
        if (!buf) { PyErr_NoMemory(); return -1; }
        memcpy(buf, &v, 8);
        *out_data = buf;
        *out_len = 8;
        *out_type = FASTDB_TYPE_INT64;
        *out_needs_free = 1;
        return 0;
    }

    if (PyFloat_Check(value)) {
        double v = PyFloat_AS_DOUBLE(value);
        char *buf = PyMem_Malloc(8);
        if (!buf) { PyErr_NoMemory(); return -1; }
        memcpy(buf, &v, 8);
        *out_data = buf;
        *out_len = 8;
        *out_type = FASTDB_TYPE_DOUBLE;
        *out_needs_free = 1;
        return 0;
    }

    if (PyDict_Check(value) || PyList_Check(value)) {
        /* JSON serialize */
        PyObject *json_mod = PyImport_ImportModule("json");
        if (!json_mod) return -1;
        PyObject *dumps = PyObject_GetAttrString(json_mod, "dumps");
        Py_DECREF(json_mod);
        if (!dumps) return -1;
        PyObject *result = PyObject_CallOneArg(dumps, value);
        Py_DECREF(dumps);
        if (!result) return -1;
        Py_ssize_t len;
        const char *utf8 = PyUnicode_AsUTF8AndSize(result, &len);
        if (!utf8) { Py_DECREF(result); return -1; }
        char *buf = PyMem_Malloc(len);
        if (!buf) { Py_DECREF(result); PyErr_NoMemory(); return -1; }
        memcpy(buf, utf8, len);
        Py_DECREF(result);
        *out_data = buf;
        *out_len = (uint32_t)len;
        *out_type = FASTDB_TYPE_JSON;
        *out_needs_free = 1;
        return 0;
    }

    PyErr_SetString(PyExc_TypeError, "Unsupported value type");
    return -1;
}

static PyObject *decode_value(const void *data, uint32_t len, fastdb_type_t type) {
    switch (type) {
    case FASTDB_TYPE_STRING:
        return PyUnicode_FromStringAndSize((const char *)data, len);
    case FASTDB_TYPE_INT64: {
        int64_t v;
        memcpy(&v, data, 8);
        return PyLong_FromLongLong(v);
    }
    case FASTDB_TYPE_UINT64: {
        uint64_t v;
        memcpy(&v, data, 8);
        return PyLong_FromUnsignedLongLong(v);
    }
    case FASTDB_TYPE_DOUBLE: {
        double v;
        memcpy(&v, data, 8);
        return PyFloat_FromDouble(v);
    }
    case FASTDB_TYPE_JSON: {
        PyObject *json_mod = PyImport_ImportModule("json");
        if (!json_mod) return NULL;
        PyObject *loads = PyObject_GetAttrString(json_mod, "loads");
        Py_DECREF(json_mod);
        if (!loads) return NULL;
        PyObject *s = PyUnicode_FromStringAndSize((const char *)data, len);
        if (!s) { Py_DECREF(loads); return NULL; }
        PyObject *result = PyObject_CallOneArg(loads, s);
        Py_DECREF(s);
        Py_DECREF(loads);
        return result;
    }
    case FASTDB_TYPE_BLOB:
    default:
        return PyBytes_FromStringAndSize((const char *)data, len);
    }
}

/* ============================================================
 * Error handling
 * ============================================================ */

static PyObject *FastDBError;
static PyObject *NotFoundError;

static int check_err(fastdb_err_t rc) {
    if (rc == FASTDB_OK) return 0;
    if (rc == FASTDB_ERR_NOT_FOUND) {
        PyErr_SetString(NotFoundError, "Key not found");
        return -1;
    }
    const char *msg;
    switch (rc) {
    case FASTDB_ERR_IO:      msg = "I/O error"; break;
    case FASTDB_ERR_CORRUPT: msg = "Database corrupt"; break;
    case FASTDB_ERR_FULL:    msg = "Database full"; break;
    case FASTDB_ERR_INVALID: msg = "Invalid argument"; break;
    case FASTDB_ERR_NOMEM:   msg = "Out of memory"; break;
    case FASTDB_ERR_LOCKED:  msg = "Database locked"; break;
    case FASTDB_ERR_WAL:     msg = "WAL error"; break;
    case FASTDB_ERR_MMAP:    msg = "Memory map error"; break;
    default:                 msg = "Unknown error"; break;
    }
    PyErr_SetString(FastDBError, msg);
    return -1;
}

/* ============================================================
 * Object lifecycle
 * ============================================================ */

static PyObject *FastDB_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    FastDBObject *self = (FastDBObject *)type->tp_alloc(type, 0);
    if (self) {
        self->db = NULL;
        self->closed = 1;
    }
    return (PyObject *)self;
}

static int FastDB_init(FastDBObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"path", "turbo", NULL};
    const char *path;
    int turbo = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|p", kwlist, &path, &turbo))
        return -1;

    uint32_t flags = turbo ? FASTDB_FLAG_TURBO : FASTDB_FLAG_DEFAULT;
    fastdb_err_t rc = fastdb_open_ex(&self->db, path, flags);
    if (rc != FASTDB_OK) {
        check_err(rc);
        return -1;
    }
    self->closed = 0;
    return 0;
}

static void FastDB_dealloc(FastDBObject *self) {
    if (self->db && !self->closed) {
        fastdb_close(self->db);
        self->db = NULL;
        self->closed = 1;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* ============================================================
 * Core CRUD methods
 * ============================================================ */

static PyObject *FastDB_put(FastDBObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"key", "value", NULL};
    PyObject *key_obj, *val_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &key_obj, &val_obj))
        return NULL;

    const char *key_data;
    Py_ssize_t key_len;

    if (PyUnicode_Check(key_obj)) {
        key_data = PyUnicode_AsUTF8AndSize(key_obj, &key_len);
        if (!key_data) return NULL;
    } else if (PyBytes_Check(key_obj)) {
        key_data = PyBytes_AS_STRING(key_obj);
        key_len = PyBytes_GET_SIZE(key_obj);
    } else {
        PyErr_SetString(PyExc_TypeError, "key must be str or bytes");
        return NULL;
    }

    const char *val_data;
    uint32_t val_len;
    fastdb_type_t val_type;
    int needs_free;

    if (encode_value(val_obj, &val_data, &val_len, &val_type, &needs_free) < 0)
        return NULL;

    fastdb_err_t rc;
    if (self->db->flags & FASTDB_FLAG_TURBO) {
        rc = fastdb_put_turbo(self->db, key_data, (uint32_t)key_len,
                              val_data, val_len, val_type);
        if (rc == FASTDB_ERR_FULL) {
            fastdb_slice_t key_slice = { .data = key_data, .len = (uint32_t)key_len };
            rc = fastdb_put(self->db, &key_slice, val_data, val_len, val_type);
        }
    } else {
        fastdb_slice_t key_slice = { .data = key_data, .len = (uint32_t)key_len };
        rc = fastdb_put(self->db, &key_slice, val_data, val_len, val_type);
    }

    if (needs_free) PyMem_Free((void *)val_data);
    if (rc != FASTDB_OK) { check_err(rc); return NULL; }
    Py_RETURN_NONE;
}

static PyObject *FastDB_get(FastDBObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"key", "default", NULL};
    PyObject *key_obj;
    PyObject *default_val = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &key_obj, &default_val))
        return NULL;

    const char *key_data;
    Py_ssize_t key_len;

    if (PyUnicode_Check(key_obj)) {
        key_data = PyUnicode_AsUTF8AndSize(key_obj, &key_len);
        if (!key_data) return NULL;
    } else if (PyBytes_Check(key_obj)) {
        key_data = PyBytes_AS_STRING(key_obj);
        key_len = PyBytes_GET_SIZE(key_obj);
    } else {
        PyErr_SetString(PyExc_TypeError, "key must be str or bytes");
        return NULL;
    }

    /* Turbo fast path */
    if (self->db->flags & FASTDB_FLAG_TURBO) {
        const void *out_data;
        uint32_t out_len;
        fastdb_type_t out_type;
        fastdb_err_t rc = fastdb_get_turbo(self->db, key_data, (uint32_t)key_len,
                                            &out_data, &out_len, &out_type);
        if (rc == FASTDB_ERR_NOT_FOUND) {
            Py_INCREF(default_val);
            return default_val;
        }
        return decode_value(out_data, out_len, out_type);
    }

    fastdb_slice_t key_slice = { .data = key_data, .len = (uint32_t)key_len };
    fastdb_value_t val;

    fastdb_err_t rc = fastdb_get(self->db, &key_slice, &val);
    if (rc == FASTDB_ERR_NOT_FOUND) {
        Py_INCREF(default_val);
        return default_val;
    }
    if (rc != FASTDB_OK) {
        check_err(rc);
        return NULL;
    }

    return decode_value(val.data.data, val.data.len, val.type);
}

static PyObject *FastDB_delete(FastDBObject *self, PyObject *args) {
    PyObject *key_obj;
    if (!PyArg_ParseTuple(args, "O", &key_obj))
        return NULL;

    const char *key_data;
    Py_ssize_t key_len;

    if (PyUnicode_Check(key_obj)) {
        key_data = PyUnicode_AsUTF8AndSize(key_obj, &key_len);
        if (!key_data) return NULL;
    } else if (PyBytes_Check(key_obj)) {
        key_data = PyBytes_AS_STRING(key_obj);
        key_len = PyBytes_GET_SIZE(key_obj);
    } else {
        PyErr_SetString(PyExc_TypeError, "key must be str or bytes");
        return NULL;
    }

    fastdb_slice_t key_slice = { .data = key_data, .len = (uint32_t)key_len };
    fastdb_err_t rc = fastdb_delete(self->db, &key_slice);
    if (rc != FASTDB_OK) {
        check_err(rc);
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *FastDB_update(FastDBObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"key", "value", NULL};
    PyObject *key_obj, *val_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &key_obj, &val_obj))
        return NULL;

    const char *key_data;
    Py_ssize_t key_len;

    if (PyUnicode_Check(key_obj)) {
        key_data = PyUnicode_AsUTF8AndSize(key_obj, &key_len);
        if (!key_data) return NULL;
    } else if (PyBytes_Check(key_obj)) {
        key_data = PyBytes_AS_STRING(key_obj);
        key_len = PyBytes_GET_SIZE(key_obj);
    } else {
        PyErr_SetString(PyExc_TypeError, "key must be str or bytes");
        return NULL;
    }

    const char *val_data;
    uint32_t val_len;
    fastdb_type_t val_type;
    int needs_free;

    if (encode_value(val_obj, &val_data, &val_len, &val_type, &needs_free) < 0)
        return NULL;

    fastdb_slice_t key_slice = { .data = key_data, .len = (uint32_t)key_len };
    fastdb_err_t rc = fastdb_update(self->db, &key_slice, val_data, val_len, val_type);

    if (needs_free) PyMem_Free((void *)val_data);

    if (rc != FASTDB_OK) {
        check_err(rc);
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *FastDB_exists(FastDBObject *self, PyObject *args) {
    PyObject *key_obj;
    if (!PyArg_ParseTuple(args, "O", &key_obj))
        return NULL;

    const char *key_data;
    Py_ssize_t key_len;

    if (PyUnicode_Check(key_obj)) {
        key_data = PyUnicode_AsUTF8AndSize(key_obj, &key_len);
        if (!key_data) return NULL;
    } else if (PyBytes_Check(key_obj)) {
        key_data = PyBytes_AS_STRING(key_obj);
        key_len = PyBytes_GET_SIZE(key_obj);
    } else {
        PyErr_SetString(PyExc_TypeError, "key must be str or bytes");
        return NULL;
    }

    fastdb_slice_t key_slice = { .data = key_data, .len = (uint32_t)key_len };
    bool result;
    fastdb_err_t rc = fastdb_exists(self->db, &key_slice, &result);
    if (rc != FASTDB_OK) {
        check_err(rc);
        return NULL;
    }
    return PyBool_FromLong(result);
}

static PyObject *FastDB_close(FastDBObject *self, PyObject *Py_UNUSED(args)) {
    if (self->db && !self->closed) {
        fastdb_close(self->db);
        self->db = NULL;
        self->closed = 1;
    }
    Py_RETURN_NONE;
}

static PyObject *FastDB_sync(FastDBObject *self, PyObject *Py_UNUSED(args)) {
    if (!self->db || self->closed) {
        PyErr_SetString(FastDBError, "Database is closed");
        return NULL;
    }
    fastdb_err_t rc = fastdb_sync(self->db);
    if (rc != FASTDB_OK) { check_err(rc); return NULL; }
    Py_RETURN_NONE;
}

static PyObject *FastDB_checkpoint(FastDBObject *self, PyObject *Py_UNUSED(args)) {
    if (!self->db || self->closed) {
        PyErr_SetString(FastDBError, "Database is closed");
        return NULL;
    }
    fastdb_err_t rc = fastdb_checkpoint(self->db);
    if (rc != FASTDB_OK) { check_err(rc); return NULL; }
    Py_RETURN_NONE;
}

static PyObject *FastDB_stats(FastDBObject *self, PyObject *Py_UNUSED(args)) {
    if (!self->db || self->closed) {
        PyErr_SetString(FastDBError, "Database is closed");
        return NULL;
    }
    fastdb_stats_t s;
    fastdb_err_t rc = fastdb_stats(self->db, &s);
    if (rc != FASTDB_OK) { check_err(rc); return NULL; }

    return Py_BuildValue("{s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:d}",
        "record_count", (unsigned long long)s.record_count,
        "data_size", (unsigned long long)s.data_size,
        "index_buckets", (unsigned long long)s.index_buckets,
        "wal_lsn", (unsigned long long)s.wal_lsn,
        "reads", (unsigned long long)s.reads,
        "writes", (unsigned long long)s.writes,
        "deletes", (unsigned long long)s.deletes,
        "scans", (unsigned long long)s.scans,
        "load_factor", s.load_factor);
}

/* Batch put */
static PyObject *FastDB_batch_put(FastDBObject *self, PyObject *args) {
    PyObject *pairs_obj;
    if (!PyArg_ParseTuple(args, "O", &pairs_obj))
        return NULL;

    if (!PyList_Check(pairs_obj)) {
        PyErr_SetString(PyExc_TypeError, "pairs must be a list");
        return NULL;
    }

    Py_ssize_t n = PyList_GET_SIZE(pairs_obj);
    if (n == 0) Py_RETURN_NONE;

    fastdb_slice_t *keys = PyMem_Malloc(n * sizeof(fastdb_slice_t));
    fastdb_slice_t *vals = PyMem_Malloc(n * sizeof(fastdb_slice_t));
    fastdb_type_t *types = PyMem_Malloc(n * sizeof(fastdb_type_t));
    char **val_bufs = PyMem_Malloc(n * sizeof(char *));
    int *needs_free = PyMem_Malloc(n * sizeof(int));

    if (!keys || !vals || !types || !val_bufs || !needs_free) {
        PyMem_Free(keys); PyMem_Free(vals); PyMem_Free(types);
        PyMem_Free(val_bufs); PyMem_Free(needs_free);
        return PyErr_NoMemory();
    }

    memset(needs_free, 0, n * sizeof(int));

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *pair = PyList_GET_ITEM(pairs_obj, i);
        if (!PyTuple_Check(pair) || PyTuple_GET_SIZE(pair) != 2) {
            PyErr_SetString(PyExc_TypeError, "Each pair must be a (key, value) tuple");
            goto batch_error;
        }

        PyObject *key_obj = PyTuple_GET_ITEM(pair, 0);
        PyObject *val_obj = PyTuple_GET_ITEM(pair, 1);

        if (PyUnicode_Check(key_obj)) {
            Py_ssize_t klen;
            const char *kdata = PyUnicode_AsUTF8AndSize(key_obj, &klen);
            if (!kdata) goto batch_error;
            keys[i].data = kdata;
            keys[i].len = (uint32_t)klen;
        } else if (PyBytes_Check(key_obj)) {
            keys[i].data = PyBytes_AS_STRING(key_obj);
            keys[i].len = (uint32_t)PyBytes_GET_SIZE(key_obj);
        } else {
            PyErr_SetString(PyExc_TypeError, "key must be str or bytes");
            goto batch_error;
        }

        const char *vdata;
        uint32_t vlen;
        fastdb_type_t vtype;
        int vfree;

        if (encode_value(val_obj, &vdata, &vlen, &vtype, &vfree) < 0)
            goto batch_error;

        vals[i].data = vdata;
        vals[i].len = vlen;
        types[i] = vtype;
        val_bufs[i] = (char *)vdata;
        needs_free[i] = vfree;
    }

    fastdb_err_t rc = fastdb_batch_put(self->db, keys, vals, types, (uint32_t)n);

    /* Cleanup */
    for (Py_ssize_t i = 0; i < n; i++) {
        if (needs_free[i]) PyMem_Free(val_bufs[i]);
    }
    PyMem_Free(keys); PyMem_Free(vals); PyMem_Free(types);
    PyMem_Free(val_bufs); PyMem_Free(needs_free);

    if (rc != FASTDB_OK) {
        check_err(rc);
        return NULL;
    }
    Py_RETURN_NONE;

batch_error:
    for (Py_ssize_t j = 0; j < n; j++) {
        if (needs_free[j]) PyMem_Free(val_bufs[j]);
    }
    PyMem_Free(keys); PyMem_Free(vals); PyMem_Free(types);
    PyMem_Free(val_bufs); PyMem_Free(needs_free);
    return NULL;
}

/* Scan callback bridge */
typedef struct {
    PyObject *callback;
    FastDBObject *self;
    int error;
} scan_ctx_t;

static bool scan_bridge(const fastdb_slice_t *key, const fastdb_value_t *value, void *user_data) {
    scan_ctx_t *ctx = (scan_ctx_t *)user_data;
    if (ctx->error) return false;

    PyObject *key_bytes = PyBytes_FromStringAndSize((const char *)key->data, key->len);
    if (!key_bytes) { ctx->error = 1; return false; }

    PyObject *val = decode_value(value->data.data, value->data.len, value->type);
    if (!val) { Py_DECREF(key_bytes); ctx->error = 1; return false; }

    PyObject *result = PyObject_CallFunctionObjArgs(ctx->callback, key_bytes, val, NULL);
    Py_DECREF(key_bytes);
    Py_DECREF(val);

    if (!result) { ctx->error = 1; return false; }
    int cont = PyObject_IsTrue(result);
    Py_DECREF(result);
    return cont > 0;
}

static PyObject *FastDB_scan(FastDBObject *self, PyObject *args) {
    PyObject *callback;
    if (!PyArg_ParseTuple(args, "O", &callback))
        return NULL;

    if (!PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "callback must be callable");
        return NULL;
    }

    scan_ctx_t ctx = { .callback = callback, .self = self, .error = 0 };
    fastdb_err_t rc = fastdb_scan(self->db, scan_bridge, &ctx);

    if (ctx.error) return NULL;
    if (rc != FASTDB_OK) { check_err(rc); return NULL; }
    Py_RETURN_NONE;
}

/* ============================================================
 * Mapping protocol for dict-like access: db[key] = val, db[key], del db[key]
 * ============================================================ */

static Py_ssize_t FastDB_mp_length(FastDBObject *self) {
    if (!self->db || self->closed) return 0;
    fastdb_stats_t s;
    if (fastdb_stats(self->db, &s) != FASTDB_OK) return 0;
    return (Py_ssize_t)s.record_count;
}

static PyObject *FastDB_mp_subscript(FastDBObject *self, PyObject *key_obj) {
    const char *key_data;
    Py_ssize_t key_len;

    if (PyUnicode_Check(key_obj)) {
        key_data = PyUnicode_AsUTF8AndSize(key_obj, &key_len);
        if (!key_data) return NULL;
    } else if (PyBytes_Check(key_obj)) {
        key_data = PyBytes_AS_STRING(key_obj);
        key_len = PyBytes_GET_SIZE(key_obj);
    } else {
        PyErr_SetString(PyExc_TypeError, "key must be str or bytes");
        return NULL;
    }

    /* Turbo fast path */
    if (self->db->flags & FASTDB_FLAG_TURBO) {
        const void *out_data;
        uint32_t out_len;
        fastdb_type_t out_type;
        fastdb_err_t rc = fastdb_get_turbo(self->db, key_data, (uint32_t)key_len,
                                            &out_data, &out_len, &out_type);
        if (rc == FASTDB_ERR_NOT_FOUND) {
            PyErr_SetObject(PyExc_KeyError, key_obj);
            return NULL;
        }
        return decode_value(out_data, out_len, out_type);
    }

    fastdb_slice_t key_slice = { .data = key_data, .len = (uint32_t)key_len };
    fastdb_value_t val;

    fastdb_err_t rc = fastdb_get(self->db, &key_slice, &val);
    if (rc == FASTDB_ERR_NOT_FOUND) {
        PyErr_SetObject(PyExc_KeyError, key_obj);
        return NULL;
    }
    if (rc != FASTDB_OK) {
        check_err(rc);
        return NULL;
    }

    return decode_value(val.data.data, val.data.len, val.type);
}

static int FastDB_mp_ass_subscript(FastDBObject *self, PyObject *key_obj, PyObject *val_obj) {
    const char *key_data;
    Py_ssize_t key_len;

    if (PyUnicode_Check(key_obj)) {
        key_data = PyUnicode_AsUTF8AndSize(key_obj, &key_len);
        if (!key_data) return -1;
    } else if (PyBytes_Check(key_obj)) {
        key_data = PyBytes_AS_STRING(key_obj);
        key_len = PyBytes_GET_SIZE(key_obj);
    } else {
        PyErr_SetString(PyExc_TypeError, "key must be str or bytes");
        return -1;
    }

    if (val_obj == NULL) {
        /* del db[key] */
        fastdb_err_t rc;
        if (self->db->flags & FASTDB_FLAG_TURBO)
            rc = fastdb_delete_turbo(self->db, key_data, (uint32_t)key_len);
        else {
            fastdb_slice_t key_slice = { .data = key_data, .len = (uint32_t)key_len };
            rc = fastdb_delete(self->db, &key_slice);
        }
        if (rc == FASTDB_ERR_NOT_FOUND) {
            PyErr_SetObject(PyExc_KeyError, key_obj);
            return -1;
        }
        if (rc != FASTDB_OK) { check_err(rc); return -1; }
        return 0;
    }

    /* db[key] = val — turbo fast path for strings */
    if ((self->db->flags & FASTDB_FLAG_TURBO) && PyUnicode_Check(val_obj)) {
        Py_ssize_t vlen;
        const char *vdata = PyUnicode_AsUTF8AndSize(val_obj, &vlen);
        if (!vdata) return -1;
        fastdb_err_t rc = fastdb_put_turbo(self->db, key_data, (uint32_t)key_len,
                                            vdata, (uint32_t)vlen, FASTDB_TYPE_STRING);
        if (rc == FASTDB_ERR_FULL) {
            /* Need to grow the map, fall through to standard path */
            fastdb_slice_t key_slice = { .data = key_data, .len = (uint32_t)key_len };
            rc = fastdb_put(self->db, &key_slice, vdata, (uint32_t)vlen, FASTDB_TYPE_STRING);
        }
        if (rc != FASTDB_OK) { check_err(rc); return -1; }
        return 0;
    }

    /* Generic path */
    const char *val_data;
    uint32_t val_len;
    fastdb_type_t val_type;
    int needs_free;

    if (encode_value(val_obj, &val_data, &val_len, &val_type, &needs_free) < 0)
        return -1;

    if (self->db->flags & FASTDB_FLAG_TURBO) {
        fastdb_err_t rc = fastdb_put_turbo(self->db, key_data, (uint32_t)key_len,
                                            val_data, val_len, val_type);
        if (rc == FASTDB_ERR_FULL) {
            fastdb_slice_t key_slice = { .data = key_data, .len = (uint32_t)key_len };
            rc = fastdb_put(self->db, &key_slice, val_data, val_len, val_type);
        }
        if (needs_free) PyMem_Free((void *)val_data);
        if (rc != FASTDB_OK) { check_err(rc); return -1; }
        return 0;
    }

    fastdb_slice_t key_slice = { .data = key_data, .len = (uint32_t)key_len };
    fastdb_err_t rc = fastdb_put(self->db, &key_slice, val_data, val_len, val_type);
    if (needs_free) PyMem_Free((void *)val_data);
    if (rc != FASTDB_OK) { check_err(rc); return -1; }
    return 0;
}

static int FastDB_sq_contains(FastDBObject *self, PyObject *key_obj) {
    const char *key_data;
    Py_ssize_t key_len;

    if (PyUnicode_Check(key_obj)) {
        key_data = PyUnicode_AsUTF8AndSize(key_obj, &key_len);
        if (!key_data) return -1;
    } else if (PyBytes_Check(key_obj)) {
        key_data = PyBytes_AS_STRING(key_obj);
        key_len = PyBytes_GET_SIZE(key_obj);
    } else {
        PyErr_SetString(PyExc_TypeError, "key must be str or bytes");
        return -1;
    }

    fastdb_slice_t key_slice = { .data = key_data, .len = (uint32_t)key_len };
    bool result;
    fastdb_err_t rc = fastdb_exists(self->db, &key_slice, &result);
    if (rc != FASTDB_OK) { check_err(rc); return -1; }
    return result ? 1 : 0;
}

/* items() / keys() / values() — collect via scan */
static PyObject *FastDB_items(FastDBObject *self, PyObject *Py_UNUSED(args)) {
    if (!self->db || self->closed) {
        PyErr_SetString(FastDBError, "Database is closed");
        return NULL;
    }
    PyObject *result = PyList_New(0);
    if (!result) return NULL;

    /* Use scan to collect all items */
    uint64_t offset = FASTDB_PAGE_SIZE;
    uint64_t end = self->db->header->data_end;

    while (offset < end && offset + sizeof(fastdb_record_t) <= end) {
        fastdb_record_t *rec = (fastdb_record_t *)((uint8_t *)self->db->data_map + offset);
        uint64_t rec_size = (sizeof(fastdb_record_t) + rec->key_len + rec->val_len + 15ULL) & ~15ULL;

        if (!(rec->flags & 0x01)) {
            PyObject *key = PyBytes_FromStringAndSize(
                (char *)rec + sizeof(fastdb_record_t), rec->key_len);
            PyObject *val = decode_value(
                (uint8_t *)rec + sizeof(fastdb_record_t) + rec->key_len,
                rec->val_len, rec->type);
            if (!key || !val) { Py_XDECREF(key); Py_XDECREF(val); Py_DECREF(result); return NULL; }
            PyObject *pair = PyTuple_Pack(2, key, val);
            Py_DECREF(key); Py_DECREF(val);
            if (!pair) { Py_DECREF(result); return NULL; }
            PyList_Append(result, pair);
            Py_DECREF(pair);
        }
        offset += rec_size;
    }
    return result;
}

static PyObject *FastDB_keys(FastDBObject *self, PyObject *Py_UNUSED(args)) {
    if (!self->db || self->closed) {
        PyErr_SetString(FastDBError, "Database is closed");
        return NULL;
    }
    PyObject *result = PyList_New(0);
    if (!result) return NULL;

    uint64_t offset = FASTDB_PAGE_SIZE;
    uint64_t end = self->db->header->data_end;

    while (offset < end && offset + sizeof(fastdb_record_t) <= end) {
        fastdb_record_t *rec = (fastdb_record_t *)((uint8_t *)self->db->data_map + offset);
        uint64_t rec_size = (sizeof(fastdb_record_t) + rec->key_len + rec->val_len + 15ULL) & ~15ULL;

        if (!(rec->flags & 0x01)) {
            PyObject *key = PyBytes_FromStringAndSize(
                (char *)rec + sizeof(fastdb_record_t), rec->key_len);
            if (!key) { Py_DECREF(result); return NULL; }
            PyList_Append(result, key);
            Py_DECREF(key);
        }
        offset += rec_size;
    }
    return result;
}

static PyObject *FastDB_values(FastDBObject *self, PyObject *Py_UNUSED(args)) {
    if (!self->db || self->closed) {
        PyErr_SetString(FastDBError, "Database is closed");
        return NULL;
    }
    PyObject *result = PyList_New(0);
    if (!result) return NULL;

    uint64_t offset = FASTDB_PAGE_SIZE;
    uint64_t end = self->db->header->data_end;

    while (offset < end && offset + sizeof(fastdb_record_t) <= end) {
        fastdb_record_t *rec = (fastdb_record_t *)((uint8_t *)self->db->data_map + offset);
        uint64_t rec_size = (sizeof(fastdb_record_t) + rec->key_len + rec->val_len + 15ULL) & ~15ULL;

        if (!(rec->flags & 0x01)) {
            PyObject *val = decode_value(
                (uint8_t *)rec + sizeof(fastdb_record_t) + rec->key_len,
                rec->val_len, rec->type);
            if (!val) { Py_DECREF(result); return NULL; }
            PyList_Append(result, val);
            Py_DECREF(val);
        }
        offset += rec_size;
    }
    return result;
}

/* Context manager */
static PyObject *FastDB_enter(FastDBObject *self, PyObject *Py_UNUSED(args)) {
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *FastDB_exit(FastDBObject *self, PyObject *args) {
    FastDB_close(self, NULL);
    Py_RETURN_FALSE;
}

/* ============================================================
 * Method table
 * ============================================================ */

static PyMethodDef FastDB_methods[] = {
    {"put",        (PyCFunction)FastDB_put,        METH_VARARGS | METH_KEYWORDS, "Insert or overwrite a key-value pair."},
    {"get",        (PyCFunction)FastDB_get,        METH_VARARGS | METH_KEYWORDS, "Retrieve a value by key."},
    {"delete",     (PyCFunction)FastDB_delete,     METH_VARARGS,                 "Delete a key."},
    {"update",     (PyCFunction)FastDB_update,     METH_VARARGS | METH_KEYWORDS, "Update an existing key."},
    {"exists",     (PyCFunction)FastDB_exists,     METH_VARARGS,                 "Check if key exists."},
    {"close",      (PyCFunction)FastDB_close,      METH_NOARGS,                  "Close the database."},
    {"sync",       (PyCFunction)FastDB_sync,       METH_NOARGS,                  "Force sync to disk."},
    {"checkpoint", (PyCFunction)FastDB_checkpoint,  METH_NOARGS,                  "Checkpoint and truncate WAL."},
    {"stats",      (PyCFunction)FastDB_stats,      METH_NOARGS,                  "Return database statistics."},
    {"batch_put",  (PyCFunction)FastDB_batch_put,  METH_VARARGS,                 "Bulk insert key-value pairs."},
    {"scan",       (PyCFunction)FastDB_scan,       METH_VARARGS,                 "Full database scan."},
    {"enable_tracking",  (PyCFunction)FastDB_enter, METH_NOARGS, "Enable tracking (no-op in C extension)."},
    {"disable_tracking", (PyCFunction)FastDB_enter, METH_NOARGS, "Disable tracking (no-op in C extension)."},
    {"items",      (PyCFunction)FastDB_items,      METH_NOARGS,                  "Return all (key, value) pairs."},
    {"keys",       (PyCFunction)FastDB_keys,       METH_NOARGS,                  "Return all keys."},
    {"values",     (PyCFunction)FastDB_values,     METH_NOARGS,                  "Return all values."},
    {"__enter__",  (PyCFunction)FastDB_enter,      METH_NOARGS,                  NULL},
    {"__exit__",   (PyCFunction)FastDB_exit,       METH_VARARGS,                 NULL},
    {NULL}
};

/* Mapping methods */
static PyMappingMethods FastDB_mapping = {
    .mp_length = (lenfunc)FastDB_mp_length,
    .mp_subscript = (binaryfunc)FastDB_mp_subscript,
    .mp_ass_subscript = (objobjargproc)FastDB_mp_ass_subscript,
};

/* Sequence methods (for 'in' operator) */
static PySequenceMethods FastDB_sequence = {
    .sq_contains = (objobjproc)FastDB_sq_contains,
};

/* ============================================================
 * Type definition
 * ============================================================ */

static PyTypeObject FastDBType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_fastdb_ext.FastDB",
    .tp_basicsize = sizeof(FastDBObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "FastDB high-performance key-value database",
    .tp_new = FastDB_new,
    .tp_init = (initproc)FastDB_init,
    .tp_dealloc = (destructor)FastDB_dealloc,
    .tp_methods = FastDB_methods,
    .tp_as_mapping = &FastDB_mapping,
    .tp_as_sequence = &FastDB_sequence,
};

/* ============================================================
 * Module-level functions
 * ============================================================ */

static PyObject *mod_open(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"path", "turbo", NULL};
    const char *path;
    int turbo = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|p", kwlist, &path, &turbo))
        return NULL;

    PyObject *arglist = Py_BuildValue("(s)", path);
    if (!arglist) return NULL;
    PyObject *kwargs = PyDict_New();
    if (!kwargs) { Py_DECREF(arglist); return NULL; }
    if (turbo) {
        PyObject *true_val = Py_True;
        Py_INCREF(true_val);
        PyDict_SetItemString(kwargs, "turbo", true_val);
        Py_DECREF(true_val);
    }
    PyObject *obj = PyObject_Call((PyObject *)&FastDBType, arglist, kwargs);
    Py_DECREF(arglist);
    Py_DECREF(kwargs);
    return obj;
}

static PyObject *mod_destroy(PyObject *self, PyObject *args) {
    const char *path;
    if (!PyArg_ParseTuple(args, "s", &path))
        return NULL;
    fastdb_destroy(path);
    Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"open",    (PyCFunction)mod_open,    METH_VARARGS | METH_KEYWORDS, "Open a FastDB database."},
    {"destroy", (PyCFunction)mod_destroy, METH_VARARGS,                 "Destroy a FastDB database."},
    {NULL}
};

/* ============================================================
 * Module definition
 * ============================================================ */

static struct PyModuleDef fastdb_module = {
    PyModuleDef_HEAD_INIT,
    "_fastdb_ext",
    "FastDB native extension module",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__fastdb_ext(void) {
    if (PyType_Ready(&FastDBType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&fastdb_module);
    if (!m) return NULL;

    Py_INCREF(&FastDBType);
    if (PyModule_AddObject(m, "FastDB", (PyObject *)&FastDBType) < 0) {
        Py_DECREF(&FastDBType);
        Py_DECREF(m);
        return NULL;
    }

    FastDBError = PyErr_NewException("_fastdb_ext.FastDBError", NULL, NULL);
    Py_XINCREF(FastDBError);
    PyModule_AddObject(m, "FastDBError", FastDBError);

    NotFoundError = PyErr_NewException("_fastdb_ext.NotFoundError", FastDBError, NULL);
    Py_XINCREF(NotFoundError);
    PyModule_AddObject(m, "NotFoundError", NotFoundError);

    /* Export constants */
    PyModule_AddIntConstant(m, "FLAG_DEFAULT", FASTDB_FLAG_DEFAULT);
    PyModule_AddIntConstant(m, "FLAG_NO_WAL",  FASTDB_FLAG_NO_WAL);
    PyModule_AddIntConstant(m, "FLAG_NO_CRC",  FASTDB_FLAG_NO_CRC);
    PyModule_AddIntConstant(m, "FLAG_NO_LOCK", FASTDB_FLAG_NO_LOCK);
    PyModule_AddIntConstant(m, "FLAG_NO_STATS", FASTDB_FLAG_NO_STATS);
    PyModule_AddIntConstant(m, "FLAG_TURBO",   FASTDB_FLAG_TURBO);

    return m;
}
