/*
 * FastDB Core Engine Implementation
 *
 * Memory-mapped storage, lock-free hash index, slab allocator,
 * WAL for durability, MVCC-lite for concurrency.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <time.h>
#include <stdatomic.h>
#include <immintrin.h>

#include "fastdb.h"

/* ============================================================
 * Internal helpers
 * ============================================================ */

#define ALIGN_UP(x, a)  (((x) + (a) - 1) & ~((a) - 1))
#define OFFSET_NONE     0ULL

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static inline uint64_t now_epoch(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec;
}

/* C fallback hash if ASM unavailable */
__attribute__((unused))
static uint64_t fastdb_hash_c(const void *data, uint32_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 0x736F6D6570736575ULL;
    for (uint32_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x100000001B3ULL;
    }
    /* splitmix64 finalizer */
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return h;
}

/* Use assembly hash if available, fall back to C */
#if defined(__x86_64__) && defined(USE_ASM)
  #define HASH(data, len) fastdb_hash_asm(data, len)
  #define MEMCMP(a, b, len) fastdb_memcmp_asm(a, b, len)
  #define CRC64(data, len) fastdb_crc64_asm(data, len)
  #define MEMCPY_NT(dst, src, len) fastdb_memcpy_nt(dst, src, len)
#else
  #define HASH(data, len) fastdb_hash_c(data, len)
  #define MEMCMP(a, b, len) memcmp(a, b, len)
  #define CRC64(data, len) fastdb_hash_c(data, len)  /* reuse hash as checksum */
  static void fastdb_memcpy_nt_c(void *dst, const void *src, size_t len) {
      memcpy(dst, src, len);
  }
  #define MEMCPY_NT(dst, src, len) fastdb_memcpy_nt_c(dst, src, len)
#endif

/* ============================================================
 * File path helpers
 * ============================================================ */

static char *make_path(const char *base, const char *ext) {
    size_t blen = strlen(base);
    size_t elen = strlen(ext);
    char *p = malloc(blen + elen + 1);
    if (!p) return NULL;
    memcpy(p, base, blen);
    memcpy(p + blen, ext, elen + 1);
    return p;
}

/* ============================================================
 * Memory map management
 * ============================================================ */

#define INITIAL_MAP_SIZE  (256ULL * 1024 * 1024)  /* 256 MB initial */
#define GROW_INCREMENT    (256ULL * 1024 * 1024)

fastdb_err_t fastdb_ensure_map_size(fastdb_t *db, size_t needed) {
    if (needed <= db->data_map_size)
        return FASTDB_OK;

    size_t new_size = ALIGN_UP(needed, GROW_INCREMENT);

    if (ftruncate(db->data_fd, new_size) < 0)
        return FASTDB_ERR_IO;

    void *new_map = mremap(db->data_map, db->data_map_size,
                           new_size, MREMAP_MAYMOVE);
    if (new_map == MAP_FAILED)
        return FASTDB_ERR_MMAP;

    db->data_map = new_map;
    db->data_map_size = new_size;
    db->header = (fastdb_header_t *)new_map;
    return FASTDB_OK;
}

/* ============================================================
 * Hash index operations
 * ============================================================ */

static fastdb_err_t index_init(fastdb_t *db, uint64_t buckets) {
    db->index_buckets = buckets;
    db->index_mask = buckets - 1;
    db->index = calloc(buckets, sizeof(uint64_t));
    if (!db->index) return FASTDB_ERR_NOMEM;
    /* Pre-touch pages to avoid page faults during operation */
    volatile uint64_t sum = 0;
    for (uint64_t i = 0; i < buckets; i += 512)
        sum += db->index[i];
    (void)sum;
    return FASTDB_OK;
}

static inline uint64_t index_bucket(fastdb_t *db, uint64_t hash) {
    return hash & db->index_mask;
}

static fastdb_err_t index_resize(fastdb_t *db) {
    uint64_t old_buckets = db->index_buckets;
    uint64_t new_buckets = old_buckets * 2;
    uint64_t *new_index = calloc(new_buckets, sizeof(uint64_t));
    if (!new_index) return FASTDB_ERR_NOMEM;

    uint64_t new_mask = new_buckets - 1;

    /* Rehash all entries */
    for (uint64_t i = 0; i < old_buckets; i++) {
        uint64_t offset = db->index[i];
        while (offset != OFFSET_NONE) {
            fastdb_record_t *rec = (fastdb_record_t *)((uint8_t *)db->data_map + offset);
            uint64_t next = rec->next_offset;

            uint64_t bucket = rec->hash & new_mask;
            rec->next_offset = new_index[bucket];
            new_index[bucket] = offset;

            offset = next;
        }
    }

    free(db->index);
    db->index = new_index;
    db->index_buckets = new_buckets;
    db->index_mask = new_mask;
    return FASTDB_OK;
}

/* ============================================================
 * WAL operations
 * ============================================================ */

static fastdb_err_t wal_write(fastdb_t *db, wal_op_t op,
                              const fastdb_slice_t *key,
                              const void *val, uint32_t val_len,
                              fastdb_type_t type) {
    fastdb_wal_entry_t entry = {0};
    entry.lsn = atomic_fetch_add(&db->wal_lsn, 1);
    entry.op = op;
    entry.key_len = key->len;
    entry.val_len = val_len;
    entry.type = type;

    size_t total = sizeof(entry) + key->len + val_len;

    /* Compute checksum over key+value */
    uint64_t ck = CRC64(key->data, key->len);
    if (val && val_len > 0) {
        ck ^= CRC64(val, val_len);
    }
    entry.checksum = ck;

    pthread_mutex_lock(&db->wal_mutex);

    /* If entry exceeds WAL buffer entirely, write directly to fd */
    if (total > FASTDB_WAL_BUFFER_SIZE) {
        /* Flush any buffered data first */
        ssize_t __attribute__((unused)) wr;
        if (db->wal_buf_pos > 0) {
            wr = write(db->wal_fd, db->wal_buf, db->wal_buf_pos);
            db->wal_buf_pos = 0;
        }
        /* Write entry header, key, value directly */
        wr = write(db->wal_fd, &entry, sizeof(entry));
        wr = write(db->wal_fd, key->data, key->len);
        if (val && val_len > 0)
            wr = write(db->wal_fd, val, val_len);
        pthread_mutex_unlock(&db->wal_mutex);
        return FASTDB_OK;
    }

    /* Flush WAL buffer if needed */
    if (db->wal_buf_pos + total > FASTDB_WAL_BUFFER_SIZE) {
        if (write(db->wal_fd, db->wal_buf, db->wal_buf_pos) < 0) {
            pthread_mutex_unlock(&db->wal_mutex);
            return FASTDB_ERR_WAL;
        }
        db->wal_buf_pos = 0;
    }

    /* Append to buffer */
    memcpy(db->wal_buf + db->wal_buf_pos, &entry, sizeof(entry));
    db->wal_buf_pos += sizeof(entry);

    memcpy(db->wal_buf + db->wal_buf_pos, key->data, key->len);
    db->wal_buf_pos += key->len;

    if (val && val_len > 0) {
        memcpy(db->wal_buf + db->wal_buf_pos, val, val_len);
        db->wal_buf_pos += val_len;
    }

    pthread_mutex_unlock(&db->wal_mutex);
    return FASTDB_OK;
}

static fastdb_err_t wal_flush(fastdb_t *db) {
    pthread_mutex_lock(&db->wal_mutex);
    if (db->wal_buf_pos == 0) {
        pthread_mutex_unlock(&db->wal_mutex);
        return FASTDB_OK;
    }
    ssize_t written = write(db->wal_fd, db->wal_buf, db->wal_buf_pos);
    db->wal_buf_pos = 0;
    pthread_mutex_unlock(&db->wal_mutex);
    if (written < 0) return FASTDB_ERR_WAL;
    return FASTDB_OK;
}

/* ============================================================
 * Database open / close / destroy
 * ============================================================ */

fastdb_err_t fastdb_open(fastdb_t **out, const char *path) {
    return fastdb_open_ex(out, path, FASTDB_FLAG_DEFAULT);
}

fastdb_err_t fastdb_open_ex(fastdb_t **out, const char *path, uint32_t flags) {
    fastdb_t *db = calloc(1, sizeof(fastdb_t));
    if (!db) return FASTDB_ERR_NOMEM;
    db->flags = flags;

    db->path = strdup(path);
    if (!db->path) { free(db); return FASTDB_ERR_NOMEM; }

    char *data_path = make_path(path, ".fdb");
    char *wal_path = make_path(path, ".wal");
    char *lock_path = make_path(path, ".lock");

    bool is_new = false;

    /* Open data file */
    db->data_fd = open(data_path, O_RDWR | O_CREAT, 0644);
    if (db->data_fd < 0) goto err_io;

    /* Advisory lock */
    db->lock_fd = open(lock_path, O_RDWR | O_CREAT, 0644);
    if (db->lock_fd >= 0) {
        flock(db->lock_fd, LOCK_EX | LOCK_NB);  /* best effort */
    }

    struct stat st;
    if (fstat(db->data_fd, &st) < 0) goto err_io;

    if (st.st_size < (off_t)FASTDB_PAGE_SIZE) {
        /* New database */
        is_new = true;
        size_t init_size = INITIAL_MAP_SIZE;
        if (ftruncate(db->data_fd, init_size) < 0) goto err_io;
        db->data_map_size = init_size;
    } else {
        db->data_map_size = ALIGN_UP(st.st_size, FASTDB_PAGE_SIZE);
    }

    /* mmap — turbo mode uses MAP_PRIVATE + POPULATE for max write speed */
    int mmap_flags = MAP_SHARED;
    if (flags & FASTDB_FLAG_TURBO) {
        mmap_flags = MAP_PRIVATE | MAP_POPULATE;
    }
    db->data_map = mmap(NULL, db->data_map_size,
                        PROT_READ | PROT_WRITE,
                        mmap_flags, db->data_fd, 0);
    if (db->data_map == MAP_FAILED) goto err_mmap;

    madvise(db->data_map, db->data_map_size, MADV_RANDOM);
    if (flags & FASTDB_FLAG_TURBO)
        madvise(db->data_map, db->data_map_size, MADV_HUGEPAGE);

    db->header = (fastdb_header_t *)db->data_map;

    if (is_new) {
        memset(db->header, 0, FASTDB_PAGE_SIZE);
        db->header->magic = FASTDB_MAGIC;
        db->header->version = FASTDB_VERSION;
        db->header->page_size = FASTDB_PAGE_SIZE;
        db->header->data_end = FASTDB_PAGE_SIZE;  /* first record starts after header */
        db->header->index_buckets = FASTDB_INITIAL_BUCKETS;
        db->header->created_ts = now_epoch();
        db->header->modified_ts = now_epoch();
    } else {
        if (db->header->magic != FASTDB_MAGIC) goto err_corrupt;
        if (db->header->data_end > db->data_map_size) goto err_corrupt;
    }

    /* Initialize hash index — turbo mode uses smaller initial size for cache efficiency */
    uint64_t init_buckets = db->header->index_buckets;
    if (is_new && (flags & FASTDB_FLAG_TURBO) && init_buckets > (1 << 18))
        init_buckets = (1 << 18);  /* 256K buckets (2MB) fits in L2/L3 */
    if (is_new)
        db->header->index_buckets = init_buckets;
    fastdb_err_t rc = index_init(db, db->header->index_buckets);
    if (rc != FASTDB_OK) goto err_nomem;

    /* Rebuild index from data if existing db */
    if (!is_new) {
        uint64_t offset = FASTDB_PAGE_SIZE;
        uint64_t end = db->header->data_end;
        while (offset < end && offset + FASTDB_RECORD_HDR_SIZE <= end) {
            fastdb_record_t *rec = (fastdb_record_t *)((uint8_t *)db->data_map + offset);
            if (!(rec->flags & 0x01)) {  /* not deleted */
                uint64_t bucket = index_bucket(db, rec->hash);
                rec->next_offset = db->index[bucket];
                db->index[bucket] = offset;
            }
            uint64_t rec_size = ALIGN_UP(
                FASTDB_RECORD_HDR_SIZE + rec->key_len + rec->val_len, 16);
            offset += rec_size;
        }
    }

    /* Open WAL */
    db->wal_fd = open(wal_path, O_RDWR | O_CREAT | O_APPEND, 0644);
    db->wal_buf = malloc(FASTDB_WAL_BUFFER_SIZE);
    if (!db->wal_buf) goto err_nomem;
    db->wal_buf_pos = 0;
    db->wal_lsn = db->header->wal_lsn;

    /* Concurrency */
    pthread_rwlock_init(&db->rw_lock, NULL);
    pthread_mutex_init(&db->wal_mutex, NULL);
    atomic_store(&db->version_counter, db->header->modified_ts);

    free(data_path);
    free(wal_path);
    free(lock_path);

    *out = db;
    return FASTDB_OK;

err_io:
    rc = FASTDB_ERR_IO; goto cleanup;
err_mmap:
    rc = FASTDB_ERR_MMAP; goto cleanup;
err_corrupt:
    rc = FASTDB_ERR_CORRUPT; goto cleanup;
err_nomem:
    rc = FASTDB_ERR_NOMEM;
cleanup:
    free(data_path);
    free(wal_path);
    free(lock_path);
    if (db->data_map && db->data_map != MAP_FAILED)
        munmap(db->data_map, db->data_map_size);
    if (db->data_fd >= 0) close(db->data_fd);
    if (db->wal_fd >= 0) close(db->wal_fd);
    if (db->lock_fd >= 0) close(db->lock_fd);
    free(db->index);
    free(db->wal_buf);
    free(db->path);
    free(db);
    return rc;
}

fastdb_err_t fastdb_close(fastdb_t *db) {
    if (!db) return FASTDB_ERR_INVALID;

    /* Flush WAL */
    if (!(db->flags & FASTDB_FLAG_NO_WAL))
        wal_flush(db);

    /* Update header */
    db->header->wal_lsn = db->wal_lsn;
    db->header->modified_ts = now_epoch();

    /* For turbo mode (MAP_PRIVATE), write data back to file */
    if (db->flags & FASTDB_FLAG_TURBO) {
        lseek(db->data_fd, 0, SEEK_SET);
        size_t data_end = db->header->data_end;
        size_t written = 0;
        while (written < data_end) {
            size_t chunk = data_end - written;
            if (chunk > (4ULL * 1024 * 1024)) chunk = 4ULL * 1024 * 1024;
            ssize_t w = write(db->data_fd, (uint8_t *)db->data_map + written, chunk);
            if (w <= 0) break;
            written += w;
        }
        fsync(db->data_fd);
    } else {
        msync(db->data_map, db->data_map_size, MS_SYNC);
    }
    munmap(db->data_map, db->data_map_size);

    if (db->lock_fd >= 0) {
        flock(db->lock_fd, LOCK_UN);
        close(db->lock_fd);
    }
    close(db->data_fd);
    close(db->wal_fd);

    pthread_rwlock_destroy(&db->rw_lock);
    pthread_mutex_destroy(&db->wal_mutex);

    free(db->index);
    free(db->wal_buf);
    free(db->path);
    free(db);
    return FASTDB_OK;
}

fastdb_err_t fastdb_destroy(const char *path) {
    char *p;
    p = make_path(path, ".fdb"); unlink(p); free(p);
    p = make_path(path, ".wal"); unlink(p); free(p);
    p = make_path(path, ".lock"); unlink(p); free(p);
    return FASTDB_OK;
}

/* ============================================================
 * Internal: find record by key
 * ============================================================ */

static fastdb_record_t *find_record(fastdb_t *db,
                                    const fastdb_slice_t *key,
                                    uint64_t hash,
                                    uint64_t *out_offset) {
    uint64_t bucket = index_bucket(db, hash);
    uint64_t offset = db->index[bucket];

    while (offset != OFFSET_NONE) {
        fastdb_record_t *rec = (fastdb_record_t *)((uint8_t *)db->data_map + offset);

        /* Fast path: compare hash first, then length, then bytes */
        if (rec->hash == hash &&
            rec->key_len == key->len &&
            !(rec->flags & 0x01)) {
            const void *rec_key = (uint8_t *)rec + FASTDB_RECORD_HDR_SIZE;
            if (MEMCMP(rec_key, key->data, key->len) == 0) {
                if (out_offset) *out_offset = offset;
                return rec;
            }
        }
        offset = rec->next_offset;
    }
    return NULL;
}

/* ============================================================
 * Internal: allocate space for a new record
 * ============================================================ */

static uint64_t alloc_record(fastdb_t *db, uint32_t key_len, uint32_t val_len) {
    uint64_t rec_size = ALIGN_UP(FASTDB_RECORD_HDR_SIZE + key_len + val_len, 16);
    uint64_t offset = db->header->data_end;
    uint64_t new_end = offset + rec_size;

    if (fastdb_ensure_map_size(db, new_end) != FASTDB_OK)
        return OFFSET_NONE;

    db->header->data_end = new_end;
    return offset;
}

/* ============================================================
 * CRUD: PUT
 * ============================================================ */

fastdb_err_t fastdb_put(fastdb_t *db,
                        const fastdb_slice_t *key,
                        const void *value, uint32_t value_len,
                        fastdb_type_t type) {
    if (!db || !key || !key->data || key->len == 0)
        return FASTDB_ERR_INVALID;
    if (key->len > FASTDB_MAX_KEY_LEN || value_len > FASTDB_MAX_VALUE_LEN)
        return FASTDB_ERR_INVALID;

    uint64_t hash = HASH(key->data, key->len);
    uint32_t flags = db->flags;

    if (!(flags & FASTDB_FLAG_NO_LOCK))
        pthread_rwlock_wrlock(&db->rw_lock);

    /* Check for existing key — update in place if possible */
    uint64_t existing_offset;
    fastdb_record_t *existing = find_record(db, key, hash, &existing_offset);
    bool is_replace = (existing != NULL);

    if (existing) {
        /* If value fits in existing slot, update in place */
        if (value_len <= existing->val_len) {
            void *val_ptr = (uint8_t *)existing + FASTDB_RECORD_HDR_SIZE + existing->key_len;
            memcpy(val_ptr, value, value_len);
            existing->val_len = value_len;
            existing->type = type;
            if (!(flags & FASTDB_FLAG_NO_STATS))
                existing->timestamp = atomic_fetch_add(&db->version_counter, 1);
            if (!(flags & FASTDB_FLAG_NO_CRC))
                existing->checksum = CRC64(key->data, key->len) ^ CRC64(value, value_len);

            if (!(flags & FASTDB_FLAG_NO_STATS))
                atomic_fetch_add(&db->stat_writes, 1);
            if (!(flags & FASTDB_FLAG_NO_LOCK))
                pthread_rwlock_unlock(&db->rw_lock);
            if (!(flags & FASTDB_FLAG_NO_WAL))
                wal_write(db, WAL_OP_PUT, key, value, value_len, type);
            return FASTDB_OK;
        }
        /* Mark old as deleted, insert new */
        existing->flags |= 0x01;
    }

    /* Allocate new record */
    uint64_t offset = alloc_record(db, key->len, value_len);
    if (offset == OFFSET_NONE) {
        if (!(flags & FASTDB_FLAG_NO_LOCK))
            pthread_rwlock_unlock(&db->rw_lock);
        return FASTDB_ERR_FULL;
    }

    fastdb_record_t *rec = (fastdb_record_t *)((uint8_t *)db->data_map + offset);
    rec->hash = hash;
    rec->key_len = key->len;
    rec->val_len = value_len;
    rec->type = type;
    rec->flags = 0;
    if (!(flags & FASTDB_FLAG_NO_STATS))
        rec->timestamp = atomic_fetch_add(&db->version_counter, 1);
    if (!(flags & FASTDB_FLAG_NO_CRC)) {
        rec->checksum = CRC64(key->data, key->len);
        if (value && value_len > 0)
            rec->checksum ^= CRC64(value, value_len);
    }

    /* Copy key */
    uint8_t *key_dst = (uint8_t *)rec + FASTDB_RECORD_HDR_SIZE;
    memcpy(key_dst, key->data, key->len);

    /* Copy value */
    if (value && value_len > 0) {
        uint8_t *val_dst = key_dst + key->len;
        memcpy(val_dst, value, value_len);
    }

    /* Insert into index */
    uint64_t bucket = index_bucket(db, hash);
    rec->next_offset = db->index[bucket];
    db->index[bucket] = offset;

    if (!is_replace)
        db->header->record_count++;
    if (!(flags & FASTDB_FLAG_NO_STATS))
        atomic_fetch_add(&db->stat_writes, 1);

    /* Resize index if load factor > 0.75 */
    double load = (double)db->header->record_count / db->index_buckets;
    if (load > 0.75) {
        index_resize(db);
        db->header->index_buckets = db->index_buckets;
    }

    if (!(flags & FASTDB_FLAG_NO_LOCK))
        pthread_rwlock_unlock(&db->rw_lock);
    if (!(flags & FASTDB_FLAG_NO_WAL))
        wal_write(db, WAL_OP_PUT, key, value, value_len, type);
    return FASTDB_OK;
}

/* ============================================================
 * CRUD: GET
 * ============================================================ */

fastdb_err_t fastdb_get(fastdb_t *db,
                        const fastdb_slice_t *key,
                        fastdb_value_t *out) {
    if (!db || !key || !key->data || key->len == 0 || !out)
        return FASTDB_ERR_INVALID;

    uint64_t hash = HASH(key->data, key->len);
    uint32_t flags = db->flags;

    if (!(flags & FASTDB_FLAG_NO_LOCK))
        pthread_rwlock_rdlock(&db->rw_lock);
    if (!(flags & FASTDB_FLAG_NO_STATS))
        atomic_fetch_add(&db->active_readers, 1);

    fastdb_record_t *rec = find_record(db, key, hash, NULL);

    if (!rec) {
        if (!(flags & FASTDB_FLAG_NO_STATS))
            atomic_fetch_sub(&db->active_readers, 1);
        if (!(flags & FASTDB_FLAG_NO_LOCK))
            pthread_rwlock_unlock(&db->rw_lock);
        if (!(flags & FASTDB_FLAG_NO_STATS))
            atomic_fetch_add(&db->stat_cache_misses, 1);
        return FASTDB_ERR_NOT_FOUND;
    }

    out->type = rec->type;
    out->data.data = (uint8_t *)rec + FASTDB_RECORD_HDR_SIZE + rec->key_len;
    out->data.len = rec->val_len;
    out->timestamp = rec->timestamp;

    if (!(flags & FASTDB_FLAG_NO_STATS)) {
        atomic_fetch_add(&db->stat_reads, 1);
        atomic_fetch_add(&db->stat_cache_hits, 1);
        atomic_fetch_sub(&db->active_readers, 1);
    }
    if (!(flags & FASTDB_FLAG_NO_LOCK))
        pthread_rwlock_unlock(&db->rw_lock);

    return FASTDB_OK;
}

/* ============================================================
 * CRUD: DELETE
 * ============================================================ */

fastdb_err_t fastdb_delete(fastdb_t *db,
                           const fastdb_slice_t *key) {
    if (!db || !key || !key->data || key->len == 0)
        return FASTDB_ERR_INVALID;

    uint64_t hash = HASH(key->data, key->len);
    uint32_t flags = db->flags;

    if (!(flags & FASTDB_FLAG_NO_LOCK))
        pthread_rwlock_wrlock(&db->rw_lock);

    uint64_t offset;
    fastdb_record_t *rec = find_record(db, key, hash, &offset);

    if (!rec) {
        if (!(flags & FASTDB_FLAG_NO_LOCK))
            pthread_rwlock_unlock(&db->rw_lock);
        return FASTDB_ERR_NOT_FOUND;
    }

    rec->flags |= 0x01;
    if (!(flags & FASTDB_FLAG_NO_STATS))
        rec->timestamp = atomic_fetch_add(&db->version_counter, 1);
    db->header->record_count--;

    if (!(flags & FASTDB_FLAG_NO_STATS))
        atomic_fetch_add(&db->stat_deletes, 1);
    if (!(flags & FASTDB_FLAG_NO_LOCK))
        pthread_rwlock_unlock(&db->rw_lock);
    if (!(flags & FASTDB_FLAG_NO_WAL))
        wal_write(db, WAL_OP_DELETE, key, NULL, 0, 0);
    return FASTDB_OK;
}

/* ============================================================
 * CRUD: UPDATE
 * ============================================================ */

fastdb_err_t fastdb_update(fastdb_t *db,
                           const fastdb_slice_t *key,
                           const void *value, uint32_t value_len,
                           fastdb_type_t type) {
    if (!db || !key || !key->data || key->len == 0)
        return FASTDB_ERR_INVALID;

    uint64_t hash = HASH(key->data, key->len);
    uint32_t flags = db->flags;

    if (!(flags & FASTDB_FLAG_NO_LOCK))
        pthread_rwlock_wrlock(&db->rw_lock);

    fastdb_record_t *rec = find_record(db, key, hash, NULL);
    if (!rec) {
        if (!(flags & FASTDB_FLAG_NO_LOCK))
            pthread_rwlock_unlock(&db->rw_lock);
        return FASTDB_ERR_NOT_FOUND;
    }

    /* In-place if fits */
    if (value_len <= rec->val_len) {
        void *val_ptr = (uint8_t *)rec + FASTDB_RECORD_HDR_SIZE + rec->key_len;
        memcpy(val_ptr, value, value_len);
        rec->val_len = value_len;
        rec->type = type;
        if (!(flags & FASTDB_FLAG_NO_STATS))
            rec->timestamp = atomic_fetch_add(&db->version_counter, 1);
        if (!(flags & FASTDB_FLAG_NO_CRC))
            rec->checksum = CRC64(key->data, key->len) ^ CRC64(value, value_len);
        if (!(flags & FASTDB_FLAG_NO_STATS))
            atomic_fetch_add(&db->stat_writes, 1);
        if (!(flags & FASTDB_FLAG_NO_LOCK))
            pthread_rwlock_unlock(&db->rw_lock);
        if (!(flags & FASTDB_FLAG_NO_WAL))
            wal_write(db, WAL_OP_PUT, key, value, value_len, type);
        return FASTDB_OK;
    }

    /* Doesn't fit: tombstone old, create new */
    rec->flags |= 0x01;

    uint64_t new_offset = alloc_record(db, key->len, value_len);
    if (new_offset == OFFSET_NONE) {
        rec->flags &= ~0x01;
        if (!(flags & FASTDB_FLAG_NO_LOCK))
            pthread_rwlock_unlock(&db->rw_lock);
        return FASTDB_ERR_FULL;
    }

    fastdb_record_t *new_rec = (fastdb_record_t *)((uint8_t *)db->data_map + new_offset);
    new_rec->hash = hash;
    new_rec->key_len = key->len;
    new_rec->val_len = value_len;
    new_rec->type = type;
    new_rec->flags = 0;
    if (!(flags & FASTDB_FLAG_NO_STATS))
        new_rec->timestamp = atomic_fetch_add(&db->version_counter, 1);
    if (!(flags & FASTDB_FLAG_NO_CRC))
        new_rec->checksum = CRC64(key->data, key->len) ^ CRC64(value, value_len);

    memcpy((uint8_t *)new_rec + FASTDB_RECORD_HDR_SIZE, key->data, key->len);
    memcpy((uint8_t *)new_rec + FASTDB_RECORD_HDR_SIZE + key->len,
              value, value_len);

    uint64_t bucket = index_bucket(db, hash);
    new_rec->next_offset = db->index[bucket];
    db->index[bucket] = new_offset;

    if (!(flags & FASTDB_FLAG_NO_STATS))
        atomic_fetch_add(&db->stat_writes, 1);
    if (!(flags & FASTDB_FLAG_NO_LOCK))
        pthread_rwlock_unlock(&db->rw_lock);
    if (!(flags & FASTDB_FLAG_NO_WAL))
        wal_write(db, WAL_OP_PUT, key, value, value_len, type);
    return FASTDB_OK;
}

/* ============================================================
 * EXISTS
 * ============================================================ */

fastdb_err_t fastdb_exists(fastdb_t *db,
                           const fastdb_slice_t *key,
                           bool *exists) {
    if (!db || !key || !exists) return FASTDB_ERR_INVALID;

    uint64_t hash = HASH(key->data, key->len);

    if (!(db->flags & FASTDB_FLAG_NO_LOCK))
        pthread_rwlock_rdlock(&db->rw_lock);
    fastdb_record_t *rec = find_record(db, key, hash, NULL);
    *exists = (rec != NULL);
    if (!(db->flags & FASTDB_FLAG_NO_LOCK))
        pthread_rwlock_unlock(&db->rw_lock);

    return FASTDB_OK;
}

/* ============================================================
 * SCAN — full database scan with callback
 * ============================================================ */

fastdb_err_t fastdb_scan(fastdb_t *db,
                         fastdb_scan_cb callback,
                         void *user_data) {
    if (!db || !callback) return FASTDB_ERR_INVALID;

    pthread_rwlock_rdlock(&db->rw_lock);
    atomic_fetch_add(&db->stat_scans, 1);

    uint64_t offset = FASTDB_PAGE_SIZE;
    uint64_t end = db->header->data_end;

    /* Prefetch hint for sequential scan */
    madvise((uint8_t *)db->data_map + offset, end - offset, MADV_SEQUENTIAL);

    while (offset < end && offset + FASTDB_RECORD_HDR_SIZE <= end) {
        fastdb_record_t *rec = (fastdb_record_t *)((uint8_t *)db->data_map + offset);
        uint64_t rec_size = ALIGN_UP(
            FASTDB_RECORD_HDR_SIZE + rec->key_len + rec->val_len, 16);

        if (!(rec->flags & 0x01)) {
            fastdb_slice_t key = {
                .data = (uint8_t *)rec + FASTDB_RECORD_HDR_SIZE,
                .len = rec->key_len,
            };
            fastdb_value_t val = {
                .type = rec->type,
                .data = {
                    .data = (uint8_t *)rec + FASTDB_RECORD_HDR_SIZE + rec->key_len,
                    .len = rec->val_len,
                },
                .timestamp = rec->timestamp,
            };

            /* Prefetch next record */
            if (offset + rec_size + FASTDB_RECORD_HDR_SIZE <= end) {
                __builtin_prefetch((uint8_t *)db->data_map + offset + rec_size, 0, 1);
            }

            if (!callback(&key, &val, user_data))
                break;
        }

        offset += rec_size;
    }

    /* Restore random access hint */
    madvise((uint8_t *)db->data_map + FASTDB_PAGE_SIZE,
            end - FASTDB_PAGE_SIZE, MADV_RANDOM);

    pthread_rwlock_unlock(&db->rw_lock);
    return FASTDB_OK;
}

/* ============================================================
 * Iterator
 * ============================================================ */

fastdb_err_t fastdb_iter_init(fastdb_iter_t *iter, fastdb_t *db) {
    if (!iter || !db) return FASTDB_ERR_INVALID;
    pthread_rwlock_rdlock(&db->rw_lock);
    iter->db = db;
    iter->offset = FASTDB_PAGE_SIZE;
    iter->end = db->header->data_end;
    iter->count = 0;
    iter->include_deleted = false;
    return FASTDB_OK;
}

fastdb_err_t fastdb_iter_next(fastdb_iter_t *iter,
                              fastdb_slice_t *key,
                              fastdb_value_t *value) {
    if (!iter || !iter->db) return FASTDB_ERR_INVALID;

    fastdb_t *db = iter->db;

    while (iter->offset < iter->end &&
           iter->offset + FASTDB_RECORD_HDR_SIZE <= iter->end) {
        fastdb_record_t *rec = (fastdb_record_t *)((uint8_t *)db->data_map + iter->offset);
        uint64_t rec_size = ALIGN_UP(
            FASTDB_RECORD_HDR_SIZE + rec->key_len + rec->val_len, 16);

        iter->offset += rec_size;

        if ((rec->flags & 0x01) && !iter->include_deleted)
            continue;

        if (key) {
            key->data = (uint8_t *)rec + FASTDB_RECORD_HDR_SIZE;
            key->len = rec->key_len;
        }
        if (value) {
            value->type = rec->type;
            value->data.data = (uint8_t *)rec + FASTDB_RECORD_HDR_SIZE + rec->key_len;
            value->data.len = rec->val_len;
            value->timestamp = rec->timestamp;
        }
        iter->count++;
        return FASTDB_OK;
    }

    /* Release read lock when iteration completes */
    pthread_rwlock_unlock(&db->rw_lock);
    iter->db = NULL;  /* mark as finished */
    return FASTDB_ERR_NOT_FOUND;  /* end of iteration */
}

/* ============================================================
 * Batch PUT — highly optimized bulk insert
 * ============================================================ */

fastdb_err_t fastdb_batch_put(fastdb_t *db,
                              const fastdb_slice_t *keys,
                              const fastdb_slice_t *values,
                              const fastdb_type_t *types,
                              uint32_t count) {
    if (!db || !keys || !values || count == 0)
        return FASTDB_ERR_INVALID;

    /* Validate all entries before taking the lock */
    for (uint32_t i = 0; i < count; i++) {
        if (!keys[i].data || keys[i].len == 0 ||
            keys[i].len > FASTDB_MAX_KEY_LEN ||
            values[i].len > FASTDB_MAX_VALUE_LEN)
            return FASTDB_ERR_INVALID;
    }

    pthread_rwlock_wrlock(&db->rw_lock);

    /* Pre-calculate total space needed */
    uint64_t total_needed = 0;
    for (uint32_t i = 0; i < count; i++) {
        total_needed += ALIGN_UP(
            FASTDB_RECORD_HDR_SIZE + keys[i].len + values[i].len, 16);
    }

    uint64_t start_offset = db->header->data_end;
    if (fastdb_ensure_map_size(db, start_offset + total_needed) != FASTDB_OK) {
        pthread_rwlock_unlock(&db->rw_lock);
        return FASTDB_ERR_FULL;
    }

    uint64_t offset = start_offset;

    for (uint32_t i = 0; i < count; i++) {
        uint64_t hash = HASH(keys[i].data, keys[i].len);
        uint64_t rec_size = ALIGN_UP(
            FASTDB_RECORD_HDR_SIZE + keys[i].len + values[i].len, 16);

        fastdb_record_t *rec = (fastdb_record_t *)((uint8_t *)db->data_map + offset);
        rec->hash = hash;
        rec->key_len = keys[i].len;
        rec->val_len = values[i].len;
        rec->type = types ? types[i] : FASTDB_TYPE_RAW;
        rec->flags = 0;
        rec->timestamp = atomic_fetch_add(&db->version_counter, 1);
        rec->checksum = CRC64(keys[i].data, keys[i].len) ^
                        CRC64(values[i].data, values[i].len);

        memcpy((uint8_t *)rec + FASTDB_RECORD_HDR_SIZE,
               keys[i].data, keys[i].len);
        memcpy((uint8_t *)rec + FASTDB_RECORD_HDR_SIZE + keys[i].len,
               values[i].data, values[i].len);

        /* Index insert */
        uint64_t bucket = index_bucket(db, hash);
        rec->next_offset = db->index[bucket];
        db->index[bucket] = offset;

        offset += rec_size;
    }

    db->header->data_end = offset;
    db->header->record_count += count;
    atomic_fetch_add(&db->stat_writes, count);

    /* Resize if needed */
    double load = (double)db->header->record_count / db->index_buckets;
    if (load > 0.75) {
        index_resize(db);
        db->header->index_buckets = db->index_buckets;
    }

    pthread_rwlock_unlock(&db->rw_lock);
    return FASTDB_OK;
}

/* ============================================================
 * WAL / Durability
 * ============================================================ */

fastdb_err_t fastdb_sync(fastdb_t *db) {
    if (!db) return FASTDB_ERR_INVALID;

    wal_flush(db);
    fsync(db->wal_fd);
    msync(db->data_map, db->header->data_end, MS_SYNC);
    fsync(db->data_fd);

    return FASTDB_OK;
}

fastdb_err_t fastdb_checkpoint(fastdb_t *db) {
    if (!db) return FASTDB_ERR_INVALID;

    fastdb_err_t rc = fastdb_sync(db);
    if (rc != FASTDB_OK) return rc;

    db->header->wal_lsn = db->wal_lsn;
    db->header->modified_ts = now_epoch();

    /* Truncate WAL */
    if (ftruncate(db->wal_fd, 0) < 0)
        return FASTDB_ERR_WAL;
    lseek(db->wal_fd, 0, SEEK_SET);

    return FASTDB_OK;
}

/* ============================================================
 * Stats
 * ============================================================ */

fastdb_err_t fastdb_stats(fastdb_t *db, fastdb_stats_t *stats) {
    if (!db || !stats) return FASTDB_ERR_INVALID;

    stats->record_count = db->header->record_count;
    stats->data_size = db->header->data_end;
    stats->index_buckets = db->index_buckets;
    stats->wal_lsn = db->wal_lsn;
    stats->reads = atomic_load(&db->stat_reads);
    stats->writes = atomic_load(&db->stat_writes);
    stats->deletes = atomic_load(&db->stat_deletes);
    stats->scans = atomic_load(&db->stat_scans);
    stats->load_factor = (double)db->header->record_count / db->index_buckets;

    return FASTDB_OK;
}
