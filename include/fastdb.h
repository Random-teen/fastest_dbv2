/*
 * FastDB - Ultra-High-Performance Database Engine
 * Core header defining the public API and internal structures.
 *
 * Architecture:
 *   - Memory-mapped file storage for zero-copy I/O
 *   - Lock-free hash index for O(1) lookups
 *   - Slab allocator for minimal fragmentation
 *   - x86-64 SIMD-accelerated hashing and scanning
 *   - WAL (Write-Ahead Log) for durability
 *   - MVCC-lite for concurrency
 */

#ifndef FASTDB_H
#define FASTDB_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Configuration constants
 * ============================================================ */

#define FASTDB_MAGIC            0x46415354444200ULL  /* "FASTDB\0\0" */
#define FASTDB_VERSION          1
#define FASTDB_PAGE_SIZE        4096
#define FASTDB_MAX_KEY_LEN      1024
#define FASTDB_MAX_VALUE_LEN    (64 * 1024 * 1024)  /* 64 MB */
#define FASTDB_INITIAL_BUCKETS  (1 << 20)            /* 1M buckets */
#define FASTDB_WAL_BUFFER_SIZE  (16 * 1024 * 1024)   /* 16 MB */
#define FASTDB_SLAB_CLASSES     32
#define FASTDB_MAX_READERS      256

/* ============================================================
 * Error codes
 * ============================================================ */

typedef enum {
    FASTDB_OK = 0,
    FASTDB_ERR_IO = -1,
    FASTDB_ERR_CORRUPT = -2,
    FASTDB_ERR_FULL = -3,
    FASTDB_ERR_NOT_FOUND = -4,
    FASTDB_ERR_EXISTS = -5,
    FASTDB_ERR_INVALID = -6,
    FASTDB_ERR_NOMEM = -7,
    FASTDB_ERR_LOCKED = -8,
    FASTDB_ERR_WAL = -9,
    FASTDB_ERR_MMAP = -10,
} fastdb_err_t;

/* ============================================================
 * Data types — flexible storage
 * ============================================================ */

typedef enum {
    FASTDB_TYPE_RAW = 0,       /* raw bytes */
    FASTDB_TYPE_INT64,
    FASTDB_TYPE_UINT64,
    FASTDB_TYPE_DOUBLE,
    FASTDB_TYPE_STRING,
    FASTDB_TYPE_BLOB,
    FASTDB_TYPE_JSON,          /* stored as string, tagged for query */
    FASTDB_TYPE_ARRAY,         /* length-prefixed array of values */
    FASTDB_TYPE_MAP,           /* nested key-value pairs */
} fastdb_type_t;

/* ============================================================
 * Key / Value structures (zero-copy friendly)
 * ============================================================ */

typedef struct {
    const void *data;
    uint32_t    len;
} fastdb_slice_t;

typedef struct {
    fastdb_type_t type;
    fastdb_slice_t data;
    uint64_t       timestamp;  /* MVCC version stamp */
} fastdb_value_t;

/* ============================================================
 * On-disk record header (packed, 48 bytes)
 * ============================================================ */

typedef struct __attribute__((packed)) {
    uint64_t hash;          /* precomputed key hash */
    uint32_t key_len;
    uint32_t val_len;
    uint8_t  type;          /* fastdb_type_t */
    uint8_t  flags;         /* 0x01 = deleted (tombstone) */
    uint8_t  _pad[2];
    uint64_t timestamp;     /* version / MVCC ts */
    uint64_t next_offset;   /* hash chain — offset in data file */
    uint64_t checksum;      /* CRC-64 of key+value */
    /* followed by: key bytes, then value bytes */
} fastdb_record_t;

#define FASTDB_RECORD_HDR_SIZE sizeof(fastdb_record_t)

/* ============================================================
 * Slab allocator
 * ============================================================ */

typedef struct {
    uint64_t slab_offset;   /* offset in mmap region */
    uint32_t slab_size;     /* size class */
    uint32_t free_count;
    uint64_t free_head;     /* offset of first free slot */
} fastdb_slab_class_t;

/* ============================================================
 * WAL entry
 * ============================================================ */

typedef enum {
    WAL_OP_PUT = 1,
    WAL_OP_DELETE = 2,
    WAL_OP_CHECKPOINT = 3,
} wal_op_t;

typedef struct __attribute__((packed)) {
    uint64_t  lsn;          /* log sequence number */
    uint8_t   op;
    uint8_t   _pad[3];
    uint32_t  key_len;
    uint32_t  val_len;
    uint8_t   type;
    uint8_t   _pad2[3];
    uint64_t  checksum;
    /* followed by key, then value */
} fastdb_wal_entry_t;

/* ============================================================
 * Database file header (first page)
 * ============================================================ */

typedef struct __attribute__((packed)) {
    uint64_t magic;
    uint32_t version;
    uint32_t page_size;
    uint64_t record_count;
    uint64_t data_end;       /* end of data region */
    uint64_t index_offset;   /* start of hash index */
    uint64_t index_buckets;
    uint64_t wal_lsn;        /* last checkpointed LSN */
    uint64_t created_ts;
    uint64_t modified_ts;
    uint8_t  reserved[4024]; /* pad to 4096 */
} fastdb_header_t;

_Static_assert(sizeof(fastdb_header_t) == FASTDB_PAGE_SIZE,
               "Header must be exactly one page");

/* ============================================================
 * Database handle
 * ============================================================ */

typedef struct {
    /* File descriptors */
    int          data_fd;
    int          wal_fd;
    int          lock_fd;

    /* Memory-mapped regions */
    void        *data_map;
    size_t       data_map_size;
    fastdb_header_t *header;

    /* In-memory hash index (offsets into data_map) */
    uint64_t    *index;          /* array of bucket head offsets */
    uint64_t     index_mask;     /* buckets - 1 */
    uint64_t     index_buckets;

    /* WAL state */
    uint8_t     *wal_buf;
    size_t       wal_buf_pos;
    _Atomic uint64_t wal_lsn;
    pthread_mutex_t  wal_mutex;  /* protects wal_buf/wal_buf_pos */

    /* Slab allocator state */
    fastdb_slab_class_t slabs[FASTDB_SLAB_CLASSES];

    /* Concurrency */
    pthread_rwlock_t   rw_lock;
    _Atomic uint64_t   version_counter;  /* MVCC timestamp */
    _Atomic uint64_t   active_readers;

    /* Path */
    char        *path;

    /* Stats */
    _Atomic uint64_t stat_reads;
    _Atomic uint64_t stat_writes;
    _Atomic uint64_t stat_deletes;
    _Atomic uint64_t stat_scans;
    _Atomic uint64_t stat_cache_hits;
    _Atomic uint64_t stat_cache_misses;
} fastdb_t;

/* ============================================================
 * Iterator for full scans
 * ============================================================ */

typedef struct {
    fastdb_t *db;
    uint64_t  offset;
    uint64_t  end;
    uint64_t  count;
    bool      include_deleted;
} fastdb_iter_t;

/* ============================================================
 * Scan callback
 * ============================================================ */

typedef bool (*fastdb_scan_cb)(const fastdb_slice_t *key,
                               const fastdb_value_t *value,
                               void *user_data);

/* ============================================================
 * Public API
 * ============================================================ */

/* Lifecycle */
fastdb_err_t fastdb_open(fastdb_t **db, const char *path);
fastdb_err_t fastdb_close(fastdb_t *db);
fastdb_err_t fastdb_destroy(const char *path);

/* CRUD */
fastdb_err_t fastdb_put(fastdb_t *db,
                        const fastdb_slice_t *key,
                        const void *value, uint32_t value_len,
                        fastdb_type_t type);

fastdb_err_t fastdb_get(fastdb_t *db,
                        const fastdb_slice_t *key,
                        fastdb_value_t *out);

fastdb_err_t fastdb_delete(fastdb_t *db,
                           const fastdb_slice_t *key);

fastdb_err_t fastdb_update(fastdb_t *db,
                           const fastdb_slice_t *key,
                           const void *value, uint32_t value_len,
                           fastdb_type_t type);

/* Existence check */
fastdb_err_t fastdb_exists(fastdb_t *db,
                           const fastdb_slice_t *key,
                           bool *exists);

/* Scan / iterate */
fastdb_err_t fastdb_scan(fastdb_t *db,
                         fastdb_scan_cb callback,
                         void *user_data);

fastdb_err_t fastdb_iter_init(fastdb_iter_t *iter, fastdb_t *db);
fastdb_err_t fastdb_iter_next(fastdb_iter_t *iter,
                              fastdb_slice_t *key,
                              fastdb_value_t *value);

/* Batch operations */
fastdb_err_t fastdb_batch_put(fastdb_t *db,
                              const fastdb_slice_t *keys,
                              const fastdb_slice_t *values,
                              const fastdb_type_t *types,
                              uint32_t count);

/* WAL / durability */
fastdb_err_t fastdb_sync(fastdb_t *db);
fastdb_err_t fastdb_checkpoint(fastdb_t *db);

/* Stats */
typedef struct {
    uint64_t record_count;
    uint64_t data_size;
    uint64_t index_buckets;
    uint64_t wal_lsn;
    uint64_t reads;
    uint64_t writes;
    uint64_t deletes;
    uint64_t scans;
    double   avg_read_ns;
    double   avg_write_ns;
    double   load_factor;
} fastdb_stats_t;

fastdb_err_t fastdb_stats(fastdb_t *db, fastdb_stats_t *stats);

/* ============================================================
 * Assembly-accelerated functions (x86-64)
 * ============================================================ */

/* Fast hash — SSE4.2 CRC32 + custom mixing */
extern uint64_t fastdb_hash_asm(const void *data, uint32_t len);

/* SIMD memcmp for key comparison */
extern int fastdb_memcmp_asm(const void *a, const void *b, uint32_t len);

/* SIMD scan — returns offset of first match or -1 */
extern int64_t fastdb_scan_asm(const void *haystack, uint64_t hay_len,
                               uint64_t target_hash);

/* CRC64 checksum */
extern uint64_t fastdb_crc64_asm(const void *data, uint32_t len);

/* Fast memcpy using non-temporal stores for large buffers */
extern void fastdb_memcpy_nt(void *dst, const void *src, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* FASTDB_H */
