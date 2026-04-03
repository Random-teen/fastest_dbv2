"""
FastDB Visualization Module

Provides rich visual analytics for database performance, storage layout,
data distribution, and operation patterns.

Usage:
    import fastdb
    from fastdb import visualize

    db = fastdb.open("/tmp/mydb")
    db.enable_tracking()

    # ... perform operations ...

    visualize.latency_histogram(db, save="latency.png")
    visualize.throughput_timeline(db, save="throughput.png")
    visualize.storage_map(db, save="storage.png")
    visualize.dashboard(db, save="dashboard.png")
"""

import os
import time
import math
from typing import Optional, Dict, List, Any, Tuple
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import numpy as np


# Custom color palette — dark theme for a database engine
COLORS = {
    'bg': '#0d1117',
    'surface': '#161b22',
    'border': '#30363d',
    'text': '#c9d1d9',
    'text_dim': '#8b949e',
    'accent': '#58a6ff',
    'green': '#3fb950',
    'red': '#f85149',
    'orange': '#d29922',
    'purple': '#bc8cff',
    'cyan': '#39d2c0',
    'pink': '#f778ba',
}

OP_COLORS = {
    'put': COLORS['green'],
    'get': COLORS['accent'],
    'get_miss': COLORS['orange'],
    'delete': COLORS['red'],
    'update': COLORS['purple'],
    'scan': COLORS['cyan'],
    'batch_put': COLORS['pink'],
}


def _setup_dark_theme():
    """Apply dark theme to matplotlib."""
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['surface'],
        'axes.edgecolor': COLORS['border'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text_dim'],
        'ytick.color': COLORS['text_dim'],
        'grid.color': COLORS['border'],
        'grid.alpha': 0.3,
        'legend.facecolor': COLORS['surface'],
        'legend.edgecolor': COLORS['border'],
        'font.family': 'monospace',
        'font.size': 10,
    })


def _save_or_show(fig, save: Optional[str] = None):
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


# ============================================================
# 1. Latency Histogram
# ============================================================

def latency_histogram(db, ops: Optional[List[str]] = None,
                      save: Optional[str] = None,
                      log_scale: bool = True) -> str:
    """
    Plot operation latency distribution as a histogram.

    Args:
        db: FastDB instance (must have tracking enabled)
        ops: List of operation types to include (default: all)
        save: Path to save the figure (default: auto-generate)
        log_scale: Use log scale for x-axis

    Returns:
        Path to saved figure
    """
    _setup_dark_theme()
    log = db.get_op_log()
    if not log:
        raise ValueError("No operation data. Call db.enable_tracking() first.")

    if ops is None:
        ops = list(set(entry['op'] for entry in log))

    fig, ax = plt.subplots(figsize=(12, 6))

    for op in ops:
        latencies = [e['latency_ns'] / 1000 for e in log if e['op'] == op]  # to µs
        if not latencies:
            continue
        color = OP_COLORS.get(op, COLORS['accent'])
        if log_scale and min(latencies) > 0:
            bins = np.logspace(np.log10(max(0.01, min(latencies))),
                               np.log10(max(latencies)), 50)
        else:
            bins = 50
        ax.hist(latencies, bins=bins, alpha=0.7, label=f"{op} (n={len(latencies)})",
                color=color, edgecolor=color, linewidth=0.5)

    if log_scale:
        ax.set_xscale('log')

    ax.set_xlabel('Latency (µs)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('FastDB Operation Latency Distribution', fontsize=14,
                 fontweight='bold', color=COLORS['accent'])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    if save is None:
        save = f"fastdb_latency_{int(time.time())}.png"

    _save_or_show(fig, save)
    return save


# ============================================================
# 2. Throughput Timeline
# ============================================================

def throughput_timeline(db, window_ms: float = 100,
                        save: Optional[str] = None) -> str:
    """
    Plot operations per second over time as a timeline.

    Args:
        db: FastDB instance
        window_ms: Aggregation window in milliseconds
        save: Path to save

    Returns:
        Path to saved figure
    """
    _setup_dark_theme()
    log = db.get_op_log()
    if not log:
        raise ValueError("No operation data.")

    # Group by time windows
    t_start = log[0]['timestamp']
    window_s = window_ms / 1000.0

    op_types = sorted(set(e['op'] for e in log))
    series = {op: defaultdict(int) for op in op_types}

    for entry in log:
        bucket = int((entry['timestamp'] - t_start) / window_s)
        series[entry['op']][bucket] += 1

    max_bucket = max(max(s.keys()) for s in series.values() if s) + 1

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(max_bucket) * window_s

    bottom = np.zeros(max_bucket)
    for op in op_types:
        y = np.array([series[op].get(i, 0) / window_s for i in range(max_bucket)])
        color = OP_COLORS.get(op, COLORS['accent'])
        ax.fill_between(x, bottom, bottom + y, alpha=0.7, label=op, color=color)
        ax.plot(x, bottom + y, color=color, linewidth=0.5, alpha=0.8)
        bottom += y

    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Operations / sec', fontsize=12, fontweight='bold')
    ax.set_title('FastDB Throughput Over Time', fontsize=14,
                 fontweight='bold', color=COLORS['accent'])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

    formatter = ticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)

    if save is None:
        save = f"fastdb_throughput_{int(time.time())}.png"

    _save_or_show(fig, save)
    return save


# ============================================================
# 3. Storage Map / Heatmap
# ============================================================

def storage_map(db, save: Optional[str] = None) -> str:
    """
    Visualize database storage layout as a heatmap grid.
    Each cell represents a region of the data file, colored by density.

    Returns:
        Path to saved figure
    """
    _setup_dark_theme()

    stats = db.stats()
    data_size = stats['data_size']
    record_count = stats['record_count']

    # Collect key sizes and value sizes
    key_sizes = []
    val_sizes = []
    type_counts = defaultdict(int)

    def _collect(k, v):
        key_sizes.append(len(k))
        if isinstance(v, str):
            val_sizes.append(len(v.encode('utf-8')))
        elif isinstance(v, bytes):
            val_sizes.append(len(v))
        elif isinstance(v, (int, float)):
            val_sizes.append(8)
        elif isinstance(v, (dict, list)):
            import json
            val_sizes.append(len(json.dumps(v).encode('utf-8')))
        else:
            val_sizes.append(0)
        type_counts[type(v).__name__] += 1
        return True

    db.scan(_collect)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Top-left: Storage regions heatmap
    ax1 = fig.add_subplot(gs[0, 0:2])
    if val_sizes:
        total_records = len(val_sizes)
        grid_size = max(1, int(math.ceil(math.sqrt(total_records))))
        grid = np.zeros((grid_size, grid_size))
        for i, vs in enumerate(val_sizes[:grid_size * grid_size]):
            row, col = divmod(i, grid_size)
            grid[row][col] = vs

        cmap = LinearSegmentedColormap.from_list('fastdb',
            ['#0d1117', '#1a3a5c', '#58a6ff', '#3fb950', '#d29922', '#f85149'])
        im = ax1.imshow(grid, cmap=cmap, aspect='auto', interpolation='nearest')
        plt.colorbar(im, ax=ax1, label='Value Size (bytes)')
    ax1.set_title('Storage Layout Heatmap', fontsize=12, fontweight='bold',
                  color=COLORS['accent'])
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')

    # Top-right: Data type distribution
    ax2 = fig.add_subplot(gs[0, 2])
    if type_counts:
        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
        colors = [COLORS['green'], COLORS['accent'], COLORS['purple'],
                  COLORS['orange'], COLORS['cyan'], COLORS['pink'],
                  COLORS['red']][:len(labels)]
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=colors, textprops={'color': COLORS['text']})
        for t in autotexts:
            t.set_fontsize(9)
    ax2.set_title('Data Type Distribution', fontsize=12, fontweight='bold',
                  color=COLORS['accent'])

    # Bottom-left: Key size distribution
    ax3 = fig.add_subplot(gs[1, 0])
    if key_sizes:
        ax3.hist(key_sizes, bins=min(50, max(1, len(set(key_sizes)))),
                 color=COLORS['accent'], alpha=0.8, edgecolor=COLORS['accent'])
    ax3.set_xlabel('Key Size (bytes)')
    ax3.set_ylabel('Count')
    ax3.set_title('Key Size Distribution', fontsize=11, fontweight='bold',
                  color=COLORS['accent'])
    ax3.grid(True, alpha=0.2)

    # Bottom-center: Value size distribution
    ax4 = fig.add_subplot(gs[1, 1])
    if val_sizes:
        ax4.hist(val_sizes, bins=min(50, max(1, len(set(val_sizes)))),
                 color=COLORS['green'], alpha=0.8, edgecolor=COLORS['green'])
    ax4.set_xlabel('Value Size (bytes)')
    ax4.set_ylabel('Count')
    ax4.set_title('Value Size Distribution', fontsize=11, fontweight='bold',
                  color=COLORS['accent'])
    ax4.grid(True, alpha=0.2)

    # Bottom-right: Stats text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    stats_text = (
        f"Records:  {record_count:,}\n"
        f"Data Size: {data_size / 1024 / 1024:.2f} MB\n"
        f"Buckets:  {stats['index_buckets']:,}\n"
        f"Load:     {stats['load_factor']:.4f}\n"
        f"Reads:    {stats['reads']:,}\n"
        f"Writes:   {stats['writes']:,}\n"
        f"Deletes:  {stats['deletes']:,}\n"
        f"Scans:    {stats['scans']:,}"
    )
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             color=COLORS['text'],
             bbox=dict(boxstyle='round', facecolor=COLORS['surface'],
                       edgecolor=COLORS['border']))
    ax5.set_title('Database Stats', fontsize=11, fontweight='bold',
                  color=COLORS['accent'])

    fig.suptitle('FastDB Storage Analysis', fontsize=16, fontweight='bold',
                 color=COLORS['accent'], y=0.98)

    if save is None:
        save = f"fastdb_storage_{int(time.time())}.png"

    _save_or_show(fig, save)
    return save


# ============================================================
# 4. Latency Percentiles
# ============================================================

def latency_percentiles(db, save: Optional[str] = None) -> str:
    """
    Plot latency percentiles (p50, p90, p95, p99, p99.9) per operation type.

    Returns:
        Path to saved figure
    """
    _setup_dark_theme()
    log = db.get_op_log()
    if not log:
        raise ValueError("No operation data.")

    ops = sorted(set(e['op'] for e in log))
    percentiles = [50, 90, 95, 99, 99.9]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(ops))
    width = 0.15

    pcolors = [COLORS['green'], COLORS['accent'], COLORS['cyan'],
               COLORS['orange'], COLORS['red']]

    for i, p in enumerate(percentiles):
        vals = []
        for op in ops:
            latencies = sorted(e['latency_ns'] / 1000
                               for e in log if e['op'] == op)
            if latencies:
                idx = min(int(len(latencies) * p / 100), len(latencies) - 1)
                vals.append(latencies[idx])
            else:
                vals.append(0)

        bars = ax.bar(x + i * width, vals, width, label=f'p{p}',
                      color=pcolors[i], alpha=0.85, edgecolor=pcolors[i])

    ax.set_xlabel('Operation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (µs)', fontsize=12, fontweight='bold')
    ax.set_title('FastDB Latency Percentiles', fontsize=14,
                 fontweight='bold', color=COLORS['accent'])
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(ops)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    if save is None:
        save = f"fastdb_percentiles_{int(time.time())}.png"

    _save_or_show(fig, save)
    return save


# ============================================================
# 5. Operation Mix / Pie
# ============================================================

def operation_mix(db, save: Optional[str] = None) -> str:
    """
    Pie chart showing the mix of operation types performed.

    Returns:
        Path to saved figure
    """
    _setup_dark_theme()
    log = db.get_op_log()
    if not log:
        raise ValueError("No operation data.")

    counts = defaultdict(int)
    for e in log:
        counts[e['op']] += 1

    fig, ax = plt.subplots(figsize=(8, 8))

    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = [OP_COLORS.get(op, COLORS['accent']) for op in labels]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, textprops={'color': COLORS['text']},
        pctdistance=0.85, startangle=90)

    for t in texts:
        t.set_fontsize(11)
        t.set_fontweight('bold')
    for t in autotexts:
        t.set_fontsize(10)

    centre_circle = plt.Circle((0, 0), 0.6, fc=COLORS['bg'])
    ax.add_artist(centre_circle)

    total = sum(sizes)
    ax.text(0, 0, f'{total:,}\nops', ha='center', va='center',
            fontsize=16, fontweight='bold', color=COLORS['accent'])

    ax.set_title('FastDB Operation Mix', fontsize=14,
                 fontweight='bold', color=COLORS['accent'])

    if save is None:
        save = f"fastdb_opmix_{int(time.time())}.png"

    _save_or_show(fig, save)
    return save


# ============================================================
# 6. Benchmark Comparison Chart
# ============================================================

def benchmark_comparison(results: Dict[str, Dict[str, float]],
                         metric: str = 'ops_per_sec',
                         save: Optional[str] = None) -> str:
    """
    Bar chart comparing FastDB against other databases.

    Args:
        results: Dict[db_name, Dict[operation, value]]
            e.g. {"FastDB": {"put": 5000000, "get": 8000000}, "SQLite": {...}}
        metric: Label for the y-axis
        save: Path to save

    Returns:
        Path to saved figure
    """
    _setup_dark_theme()

    db_names = list(results.keys())
    if not db_names:
        raise ValueError("No benchmark results")

    ops = list(results[db_names[0]].keys())

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(ops))
    total_bars = len(db_names)
    width = 0.8 / total_bars

    db_colors = [COLORS['accent'], COLORS['green'], COLORS['orange'],
                 COLORS['purple'], COLORS['cyan'], COLORS['pink'], COLORS['red']]

    for i, db_name in enumerate(db_names):
        vals = [results[db_name].get(op, 0) for op in ops]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      label=db_name, color=db_colors[i % len(db_colors)],
                      alpha=0.9, edgecolor=db_colors[i % len(db_colors)],
                      linewidth=0.5)

        # Add value labels on top
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:,.0f}', ha='center', va='bottom',
                        fontsize=7, color=COLORS['text_dim'], rotation=45)

    ax.set_xlabel('Operation', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title('FastDB vs Other Databases — Performance Comparison',
                 fontsize=14, fontweight='bold', color=COLORS['accent'])
    ax.set_xticks(x)
    ax.set_xticklabels(ops, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    formatter = ticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)

    if save is None:
        save = f"fastdb_benchmark_{int(time.time())}.png"

    _save_or_show(fig, save)
    return save


# ============================================================
# 7. Full Dashboard
# ============================================================

def dashboard(db, save: Optional[str] = None) -> str:
    """
    Comprehensive dashboard combining latency, throughput, storage, and stats.

    Returns:
        Path to saved figure
    """
    _setup_dark_theme()
    log = db.get_op_log()
    stats_data = db.stats()

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ---- 1. Latency histogram (top-left, wide) ----
    ax1 = fig.add_subplot(gs[0, 0:2])
    if log:
        ops = sorted(set(e['op'] for e in log))
        for op in ops:
            latencies = [e['latency_ns'] / 1000 for e in log if e['op'] == op]
            if latencies and min(latencies) > 0:
                bins = np.logspace(np.log10(max(0.01, min(latencies))),
                                   np.log10(max(latencies)), 40)
                ax1.hist(latencies, bins=bins, alpha=0.7, label=op,
                         color=OP_COLORS.get(op, COLORS['accent']))
        ax1.set_xscale('log')
        ax1.legend(fontsize=8)
    ax1.set_xlabel('Latency (µs)')
    ax1.set_ylabel('Count')
    ax1.set_title('Latency Distribution', fontsize=12, fontweight='bold',
                  color=COLORS['accent'])
    ax1.grid(True, alpha=0.2)

    # ---- 2. Stats panel (top-right) ----
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    stats_text = (
        f"{'FastDB Dashboard':^30}\n"
        f"{'─' * 30}\n"
        f"Records:     {stats_data['record_count']:>12,}\n"
        f"Data Size:   {stats_data['data_size_mb']:>10.2f} MB\n"
        f"Buckets:     {stats_data['index_buckets']:>12,}\n"
        f"Load Factor: {stats_data['load_factor']:>12.4f}\n"
        f"{'─' * 30}\n"
        f"Total Reads: {stats_data['reads']:>12,}\n"
        f"Total Writes:{stats_data['writes']:>12,}\n"
        f"Total Deletes:{stats_data['deletes']:>11,}\n"
        f"Total Scans: {stats_data['scans']:>12,}\n"
    )
    if log:
        all_latencies = [e['latency_ns'] / 1000 for e in log]
        stats_text += (
            f"{'─' * 30}\n"
            f"Avg Latency: {np.mean(all_latencies):>10.1f} µs\n"
            f"P50 Latency: {np.percentile(all_latencies, 50):>10.1f} µs\n"
            f"P99 Latency: {np.percentile(all_latencies, 99):>10.1f} µs\n"
        )

    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             color=COLORS['text'],
             bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['surface'],
                       edgecolor=COLORS['border'], linewidth=1.5))

    # ---- 3. Throughput timeline (middle, full width) ----
    ax3 = fig.add_subplot(gs[1, :])
    if log:
        t_start = log[0]['timestamp']
        window_s = 0.05
        op_types = sorted(set(e['op'] for e in log))
        series = {op: defaultdict(int) for op in op_types}
        for entry in log:
            bucket = int((entry['timestamp'] - t_start) / window_s)
            series[entry['op']][bucket] += 1
        max_bucket = max(max(s.keys()) for s in series.values() if s) + 1
        x = np.arange(max_bucket) * window_s
        bottom = np.zeros(max_bucket)
        for op in op_types:
            y = np.array([series[op].get(i, 0) / window_s
                          for i in range(max_bucket)])
            color = OP_COLORS.get(op, COLORS['accent'])
            ax3.fill_between(x, bottom, bottom + y, alpha=0.7,
                             label=op, color=color)
            bottom += y
        ax3.legend(fontsize=8, loc='upper right')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Ops/sec')
    ax3.set_title('Throughput Over Time', fontsize=12, fontweight='bold',
                  color=COLORS['accent'])
    ax3.grid(True, alpha=0.2)

    # ---- 4. Percentiles bar (bottom-left) ----
    ax4 = fig.add_subplot(gs[2, 0])
    if log:
        ops = sorted(set(e['op'] for e in log))
        percentiles = [50, 95, 99]
        x = np.arange(len(ops))
        width = 0.25
        pcolors = [COLORS['green'], COLORS['orange'], COLORS['red']]
        for i, p in enumerate(percentiles):
            vals = []
            for op in ops:
                lat = sorted(e['latency_ns'] / 1000 for e in log if e['op'] == op)
                if lat:
                    idx = min(int(len(lat) * p / 100), len(lat) - 1)
                    vals.append(lat[idx])
                else:
                    vals.append(0)
            ax4.bar(x + i * width, vals, width, label=f'p{p}',
                    color=pcolors[i], alpha=0.85)
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(ops, fontsize=8, rotation=30)
        ax4.legend(fontsize=8)
    ax4.set_ylabel('Latency (µs)')
    ax4.set_title('Latency Percentiles', fontsize=11, fontweight='bold',
                  color=COLORS['accent'])
    ax4.grid(True, alpha=0.2, axis='y')

    # ---- 5. Operation mix donut (bottom-center) ----
    ax5 = fig.add_subplot(gs[2, 1])
    if log:
        counts = defaultdict(int)
        for e in log:
            counts[e['op']] += 1
        labels = list(counts.keys())
        sizes = list(counts.values())
        colors = [OP_COLORS.get(op, COLORS['accent']) for op in labels]
        wedges, texts, autotexts = ax5.pie(
            sizes, labels=labels, autopct='%1.0f%%',
            colors=colors, textprops={'color': COLORS['text'], 'fontsize': 8},
            pctdistance=0.8, startangle=90)
        centre = plt.Circle((0, 0), 0.5, fc=COLORS['bg'])
        ax5.add_artist(centre)
        ax5.text(0, 0, f'{sum(sizes):,}', ha='center', va='center',
                fontsize=12, fontweight='bold', color=COLORS['accent'])
    ax5.set_title('Operation Mix', fontsize=11, fontweight='bold',
                  color=COLORS['accent'])

    # ---- 6. Cumulative ops (bottom-right) ----
    ax6 = fig.add_subplot(gs[2, 2])
    if log:
        t_start = log[0]['timestamp']
        times = [(e['timestamp'] - t_start) for e in log]
        cumulative = np.arange(1, len(times) + 1)
        ax6.plot(times, cumulative, color=COLORS['accent'], linewidth=1.5)
        ax6.fill_between(times, cumulative, alpha=0.15, color=COLORS['accent'])
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Total Operations')
    ax6.set_title('Cumulative Operations', fontsize=11, fontweight='bold',
                  color=COLORS['accent'])
    ax6.grid(True, alpha=0.2)

    fig.suptitle('FastDB Performance Dashboard', fontsize=18,
                 fontweight='bold', color=COLORS['accent'], y=0.99)

    if save is None:
        save = f"fastdb_dashboard_{int(time.time())}.png"

    _save_or_show(fig, save)
    return save


# ============================================================
# 8. Key Distribution Visualization
# ============================================================

def key_distribution(db, save: Optional[str] = None) -> str:
    """
    Visualize how keys are distributed across hash buckets.

    Returns:
        Path to saved figure
    """
    _setup_dark_theme()

    # Collect key hashes
    key_hashes = []
    def _collect(k, v):
        h = hash(k) & 0xFFFFFFFF
        key_hashes.append(h)
        return True

    db.scan(_collect)

    if not key_hashes:
        raise ValueError("Database is empty")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: hash distribution histogram
    ax1 = axes[0]
    ax1.hist(key_hashes, bins=100, color=COLORS['accent'], alpha=0.8,
             edgecolor=COLORS['accent'], linewidth=0.3)
    ax1.set_xlabel('Hash Value')
    ax1.set_ylabel('Count')
    ax1.set_title('Key Hash Distribution', fontsize=12, fontweight='bold',
                  color=COLORS['accent'])
    ax1.grid(True, alpha=0.2)

    # Right: bucket occupancy (simulated)
    ax2 = axes[1]
    n_buckets = min(256, db.stats()['index_buckets'])
    bucket_counts = [0] * n_buckets
    for h in key_hashes:
        bucket_counts[h % n_buckets] += 1

    colors = [COLORS['green'] if c <= np.mean(bucket_counts) * 1.5
              else COLORS['orange'] if c <= np.mean(bucket_counts) * 3
              else COLORS['red'] for c in bucket_counts]

    ax2.bar(range(n_buckets), bucket_counts, color=colors, alpha=0.8, width=1.0)
    ax2.axhline(y=np.mean(bucket_counts), color=COLORS['accent'],
                linestyle='--', linewidth=1, label=f'Mean: {np.mean(bucket_counts):.1f}')
    ax2.set_xlabel('Bucket Index')
    ax2.set_ylabel('Keys per Bucket')
    ax2.set_title('Hash Bucket Occupancy', fontsize=12, fontweight='bold',
                  color=COLORS['accent'])
    ax2.legend()
    ax2.grid(True, alpha=0.2, axis='y')

    fig.suptitle('FastDB Key Distribution Analysis', fontsize=14,
                 fontweight='bold', color=COLORS['accent'])

    if save is None:
        save = f"fastdb_keydist_{int(time.time())}.png"

    _save_or_show(fig, save)
    return save


# ============================================================
# 9. Value Type Treemap (simplified)
# ============================================================

def data_overview(db, save: Optional[str] = None) -> str:
    """
    Overview of stored data: types, sizes, and sample keys.

    Returns:
        Path to saved figure
    """
    _setup_dark_theme()

    type_info = defaultdict(lambda: {'count': 0, 'total_size': 0, 'sample_keys': []})

    def _collect(k, v):
        tname = type(v).__name__
        info = type_info[tname]
        info['count'] += 1
        if isinstance(v, (str, bytes)):
            info['total_size'] += len(v)
        elif isinstance(v, (int, float)):
            info['total_size'] += 8
        if len(info['sample_keys']) < 3:
            try:
                info['sample_keys'].append(k.decode('utf-8')[:30])
            except Exception:
                info['sample_keys'].append(repr(k)[:30])
        return True

    db.scan(_collect)

    if not type_info:
        raise ValueError("Database is empty")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: size by type
    ax1 = axes[0]
    types = list(type_info.keys())
    sizes = [type_info[t]['total_size'] / 1024 for t in types]
    counts = [type_info[t]['count'] for t in types]
    colors = [COLORS['green'], COLORS['accent'], COLORS['purple'],
              COLORS['orange'], COLORS['cyan']][:len(types)]

    bars = ax1.barh(types, sizes, color=colors, alpha=0.85)
    ax1.set_xlabel('Total Size (KB)')
    ax1.set_title('Storage by Data Type', fontsize=12, fontweight='bold',
                  color=COLORS['accent'])
    ax1.grid(True, alpha=0.2, axis='x')

    # Right: record count by type
    ax2 = axes[1]
    bars2 = ax2.barh(types, counts, color=colors, alpha=0.85)
    ax2.set_xlabel('Record Count')
    ax2.set_title('Records by Data Type', fontsize=12, fontweight='bold',
                  color=COLORS['accent'])
    ax2.grid(True, alpha=0.2, axis='x')

    fig.suptitle('FastDB Data Overview', fontsize=14,
                 fontweight='bold', color=COLORS['accent'])

    if save is None:
        save = f"fastdb_overview_{int(time.time())}.png"

    _save_or_show(fig, save)
    return save
