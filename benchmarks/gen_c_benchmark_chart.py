#!/usr/bin/env python3
"""Generate a chart from C benchmark results showing raw engine performance."""

import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from fastdb import visualize

SCREENSHOTS_DIR = os.environ.get('TWILL_SCREENSHOTS_DIR', '/root/.screenshots')

# Load C benchmark results
with open('/tmp/fastdb_bench_results.json') as f:
    data = json.load(f)

results = data['results']

# Compare C-level FastDB performance
comparison = {
    'FastDB (C engine)': {
        'seq_insert': results['sequential_insert'],
        'seq_read': results['sequential_read'],
        'random_read': results['random_read'],
        'update': results['update'],
        'delete': results['delete'],
        'full_scan': results['full_scan'],
        'batch_insert': results['batch_insert'],
    },
}

chart_path = os.path.join(SCREENSHOTS_DIR, "fastdb_c_benchmark.png")
visualize.benchmark_comparison(comparison, metric='ops_per_sec', save=chart_path)
print(f"Chart saved: {chart_path}")
