/*
 * FastDB Comprehensive Benchmark Suite
 *
 * Tests: sequential insert, random insert, point lookup, random lookup,
 *        update, delete, full scan, batch insert.
 * Outputs JSON for visualization and comparison.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include "fastdb.h"

#define BENCH_COUNT     1000000
#define BATCH_SIZE      10000
#define SCAN_RECORDS    500000
#define VALUE_SIZE      100
#define KEY_PREFIX      "bench_key_"
#define WARMUP_COUNT    10000

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* Simple xorshift PRNG — fast and good enough for benchmarks */
static uint64_t rng_state = 0x12345678ABCDEF01ULL;
static inline uint64_t xorshift64(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

typedef struct {
    double min_ns;
    double max_ns;
    double avg_ns;
    double p50_ns;
    double p95_ns;
    double p99_ns;
    double p999_ns;
    double total_s;
    double ops_per_sec;
    uint64_t count;
} bench_result_t;

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void compute_stats(double *latencies, uint64_t count,
                           bench_result_t *result) {
    qsort(latencies, count, sizeof(double), cmp_double);

    double sum = 0;
    for (uint64_t i = 0; i < count; i++) sum += latencies[i];

    result->count = count;
    result->min_ns = latencies[0];
    result->max_ns = latencies[count - 1];
    result->avg_ns = sum / count;
    result->p50_ns = latencies[(uint64_t)(count * 0.50)];
    result->p95_ns = latencies[(uint64_t)(count * 0.95)];
    result->p99_ns = latencies[(uint64_t)(count * 0.99)];
    result->p999_ns = latencies[(uint64_t)(count * 0.999)];
    result->total_s = sum / 1e9;
    result->ops_per_sec = count / (sum / 1e9);
}

static void print_result(const char *name, bench_result_t *r) {
    printf("  %-22s  %10.0f ops/s  avg=%6.0f ns  p50=%6.0f  p95=%6.0f  "
           "p99=%6.0f  p99.9=%7.0f  (n=%lu)\n",
           name, r->ops_per_sec, r->avg_ns, r->p50_ns, r->p95_ns,
           r->p99_ns, r->p999_ns, r->count);
}

static char *make_key(char *buf, uint64_t i) {
    sprintf(buf, KEY_PREFIX "%020lu", i);
    return buf;
}

static void make_value(char *buf, int size) {
    for (int i = 0; i < size; i++)
        buf[i] = 'A' + (i % 26);
}

/* ============================================================ */

int main(int argc, char **argv) {
    const char *db_path = "/tmp/fastdb_bench";
    uint64_t n = BENCH_COUNT;

    if (argc > 1) n = atol(argv[1]);

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║                  FastDB Benchmark Suite v1.0                    ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Records: %-10lu  Value size: %-5d bytes                   ║\n", n, VALUE_SIZE);
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    /* Clean up previous runs */
    fastdb_destroy(db_path);

    fastdb_t *db;
    fastdb_err_t rc = fastdb_open(&db, db_path);
    if (rc != FASTDB_OK) {
        fprintf(stderr, "Failed to open database: %d\n", rc);
        return 1;
    }

    char key_buf[128];
    char val_buf[VALUE_SIZE + 1];
    make_value(val_buf, VALUE_SIZE);

    double *latencies = malloc(n * sizeof(double));
    if (!latencies) {
        fprintf(stderr, "Failed to allocate latency buffer\n");
        return 1;
    }

    bench_result_t result;

    /* ---- Warmup ---- */
    printf("  Warming up (%d ops)...\n", WARMUP_COUNT);
    for (uint64_t i = 0; i < WARMUP_COUNT; i++) {
        make_key(key_buf, i + n + 1000000);
        fastdb_slice_t ks = {key_buf, strlen(key_buf)};
        fastdb_put(db, &ks, val_buf, VALUE_SIZE, FASTDB_TYPE_STRING);
    }
    /* Delete warmup data */
    for (uint64_t i = 0; i < WARMUP_COUNT; i++) {
        make_key(key_buf, i + n + 1000000);
        fastdb_slice_t ks = {key_buf, strlen(key_buf)};
        fastdb_delete(db, &ks);
    }

    printf("\n  ── Results ────────────────────────────────────────────────────\n\n");

    /* ============================================================
     * 1. Sequential Insert
     * ============================================================ */
    for (uint64_t i = 0; i < n; i++) {
        make_key(key_buf, i);
        fastdb_slice_t ks = {key_buf, strlen(key_buf)};
        uint64_t t0 = now_ns();
        fastdb_put(db, &ks, val_buf, VALUE_SIZE, FASTDB_TYPE_STRING);
        latencies[i] = (double)(now_ns() - t0);
    }
    compute_stats(latencies, n, &result);
    print_result("Sequential Insert", &result);
    double seq_insert_ops = result.ops_per_sec;

    /* ============================================================
     * 2. Sequential Read
     * ============================================================ */
    for (uint64_t i = 0; i < n; i++) {
        make_key(key_buf, i);
        fastdb_slice_t ks = {key_buf, strlen(key_buf)};
        fastdb_value_t val;
        uint64_t t0 = now_ns();
        fastdb_get(db, &ks, &val);
        latencies[i] = (double)(now_ns() - t0);
    }
    compute_stats(latencies, n, &result);
    print_result("Sequential Read", &result);
    double seq_read_ops = result.ops_per_sec;

    /* ============================================================
     * 3. Random Read
     * ============================================================ */
    for (uint64_t i = 0; i < n; i++) {
        uint64_t idx = xorshift64() % n;
        make_key(key_buf, idx);
        fastdb_slice_t ks = {key_buf, strlen(key_buf)};
        fastdb_value_t val;
        uint64_t t0 = now_ns();
        fastdb_get(db, &ks, &val);
        latencies[i] = (double)(now_ns() - t0);
    }
    compute_stats(latencies, n, &result);
    print_result("Random Read", &result);
    double rand_read_ops = result.ops_per_sec;

    /* ============================================================
     * 4. Update (in-place, same size)
     * ============================================================ */
    char new_val[VALUE_SIZE];
    memset(new_val, 'Z', VALUE_SIZE);

    for (uint64_t i = 0; i < n; i++) {
        make_key(key_buf, i);
        fastdb_slice_t ks = {key_buf, strlen(key_buf)};
        uint64_t t0 = now_ns();
        fastdb_update(db, &ks, new_val, VALUE_SIZE, FASTDB_TYPE_STRING);
        latencies[i] = (double)(now_ns() - t0);
    }
    compute_stats(latencies, n, &result);
    print_result("Update (in-place)", &result);
    double update_ops = result.ops_per_sec;

    /* ============================================================
     * 5. Delete
     * ============================================================ */
    /* Only delete half so we can still test scan */
    uint64_t del_count = n / 2;
    for (uint64_t i = 0; i < del_count; i++) {
        uint64_t idx = i * 2 + 1;  /* delete odd-indexed keys */
        make_key(key_buf, idx);
        fastdb_slice_t ks = {key_buf, strlen(key_buf)};
        uint64_t t0 = now_ns();
        fastdb_delete(db, &ks);
        latencies[i] = (double)(now_ns() - t0);
    }
    compute_stats(latencies, del_count, &result);
    print_result("Delete", &result);
    double delete_ops = result.ops_per_sec;

    /* ============================================================
     * 6. Full Scan
     * ============================================================ */
    volatile uint64_t scan_count = 0;
    uint64_t scan_start = now_ns();

    fastdb_iter_t iter;
    fastdb_iter_init(&iter, db);
    fastdb_slice_t sk;
    fastdb_value_t sv;
    while (fastdb_iter_next(&iter, &sk, &sv) == FASTDB_OK) {
        scan_count++;
    }

    uint64_t scan_elapsed = now_ns() - scan_start;
    double scan_ops = (double)scan_count / ((double)scan_elapsed / 1e9);
    printf("  %-22s  %10.0f ops/s  total=%6.3f s  records=%lu\n",
           "Full Scan", scan_ops, (double)scan_elapsed / 1e9, scan_count);

    /* ============================================================
     * 7. Batch Insert
     * ============================================================ */
    /* Close and reopen for clean batch test */
    fastdb_close(db);
    fastdb_destroy(db_path);
    fastdb_open(&db, db_path);

    uint64_t batch_n = n;
    uint64_t num_batches = batch_n / BATCH_SIZE;

    /* Pre-allocate batch arrays */
    fastdb_slice_t *batch_keys = malloc(BATCH_SIZE * sizeof(fastdb_slice_t));
    fastdb_slice_t *batch_vals = malloc(BATCH_SIZE * sizeof(fastdb_slice_t));
    fastdb_type_t *batch_types = malloc(BATCH_SIZE * sizeof(fastdb_type_t));
    char **key_ptrs = malloc(BATCH_SIZE * sizeof(char *));

    for (int i = 0; i < BATCH_SIZE; i++) {
        key_ptrs[i] = malloc(128);
        batch_vals[i].data = val_buf;
        batch_vals[i].len = VALUE_SIZE;
        batch_types[i] = FASTDB_TYPE_STRING;
    }

    uint64_t batch_total_ns = 0;
    for (uint64_t b = 0; b < num_batches; b++) {
        for (int i = 0; i < BATCH_SIZE; i++) {
            make_key(key_ptrs[i], b * BATCH_SIZE + i);
            batch_keys[i].data = key_ptrs[i];
            batch_keys[i].len = strlen(key_ptrs[i]);
        }

        uint64_t t0 = now_ns();
        fastdb_batch_put(db, batch_keys, batch_vals, batch_types, BATCH_SIZE);
        batch_total_ns += now_ns() - t0;
    }

    double batch_ops = (double)batch_n / ((double)batch_total_ns / 1e9);
    printf("  %-22s  %10.0f ops/s  total=%6.3f s  (batch_size=%d)\n",
           "Batch Insert", batch_ops, (double)batch_total_ns / 1e9, BATCH_SIZE);

    /* ============================================================
     * Summary
     * ============================================================ */
    printf("\n  ── Summary ───────────────────────────────────────────────────\n\n");

    fastdb_stats_t stats;
    fastdb_stats(db, &stats);
    printf("  Records:      %lu\n", stats.record_count);
    printf("  Data size:    %.2f MB\n", (double)stats.data_size / 1024 / 1024);
    printf("  Index load:   %.4f\n", stats.load_factor);
    printf("  Total writes: %lu\n", stats.writes);
    printf("  Total reads:  %lu\n", stats.reads);

    /* ============================================================
     * JSON output for visualization
     * ============================================================ */
    FILE *fp = fopen("/tmp/fastdb_bench_results.json", "w");
    if (fp) {
        fprintf(fp, "{\n");
        fprintf(fp, "  \"engine\": \"FastDB\",\n");
        fprintf(fp, "  \"record_count\": %lu,\n", n);
        fprintf(fp, "  \"value_size\": %d,\n", VALUE_SIZE);
        fprintf(fp, "  \"results\": {\n");
        fprintf(fp, "    \"sequential_insert\": %.0f,\n", seq_insert_ops);
        fprintf(fp, "    \"sequential_read\": %.0f,\n", seq_read_ops);
        fprintf(fp, "    \"random_read\": %.0f,\n", rand_read_ops);
        fprintf(fp, "    \"update\": %.0f,\n", update_ops);
        fprintf(fp, "    \"delete\": %.0f,\n", delete_ops);
        fprintf(fp, "    \"full_scan\": %.0f,\n", scan_ops);
        fprintf(fp, "    \"batch_insert\": %.0f\n", batch_ops);
        fprintf(fp, "  }\n");
        fprintf(fp, "}\n");
        fclose(fp);
        printf("\n  Results saved to /tmp/fastdb_bench_results.json\n");
    }

    /* Cleanup */
    for (int i = 0; i < BATCH_SIZE; i++) free(key_ptrs[i]);
    free(key_ptrs);
    free(batch_keys);
    free(batch_vals);
    free(batch_types);
    free(latencies);

    fastdb_close(db);
    fastdb_destroy(db_path);

    printf("\n  Done.\n\n");
    return 0;
}
