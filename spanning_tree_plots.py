"""Wasserstein trace plots for RevReCom paper."""
import json
import numpy as np
import requests
import click
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from client import ChainDBClient
from typing import List, Dict, Optional
from collections import Counter
from scipy.stats import wasserstein_distance
from tqdm import tqdm

FIG_SIZES = {
    'wasserstein': (10, 6),
    'boxplots': (16, 8)
}


@click.command()
@click.option('--chain-data', multiple=True)
@click.option('--chain-label', multiple=True)
@click.option('--end-step', type=int)
@click.option('--fig-title')
@click.option('--fig-file', required=True)
@click.option('--fig-type', default='wasserstein')
@click.option('--wasserstein-resolution', default=1000, type=int)
def main(chain_data, chain_label, end_step, fig_title, fig_file,
         fig_type, wasserstein_resolution):
    print(chain_data)
    print(chain_label)
    shares = {}
    weights = {}
    for filename, label in zip(chain_data, chain_label):
        chain_shares, chain_weights, chain_meta = load_from_jsonl(
            filename, end_step)
        print(chain_meta)
        shares[label] = chain_shares
        weights[label] = chain_weights

    fig, ax = plt.subplots(figsize=FIG_SIZES[fig_type])
    if fig_type == 'wasserstein':
        for outer_idx, outer_label in enumerate(chain_label):
            for inner_idx, inner_label in enumerate(chain_label):
                if inner_idx > outer_idx:
                    xs, dists = wasserstein_trace(shares[outer_label],
                                                  shares[inner_label],
                                                  weights[outer_label],
                                                  weights[inner_label],
                                                  wasserstein_resolution)
                    ax.plot(xs,
                            dists,
                            label=f'{outer_label} vs. {inner_label}')
                    ax.set_xlabel('Unique steps')
                    ax.set_ylabel('Distance')
    elif fig_type == 'boxplots':
        stats = [
            raw_to_hists(shares[label], weights[label])
            for label in chain_label
        ]
        shares_boxplot_by_chain(stats,
                                chain_label,
                                ax,
                                min_col=None,
                                max_col=None)
        ax.set_xlabel('District')
        ax.set_ylabel('log(sp)')
    ax.legend()
    ax.set_title(fig_title)
    plt.savefig(fig_file, bbox_inches='tight')
    plt.close(fig)


def wasserstein_trace(shares1, shares2, weights1, weights2, resolution):
    n_districts = len(shares1[0])
    assert len(shares1[0]) == len(shares2[0])
    state1 = np.zeros(n_districts)
    state2 = np.zeros(n_districts)
    xticks = []
    trace = []
    hist1 = [Counter() for _ in range(n_districts)]
    hist2 = [Counter() for _ in range(n_districts)]
    for step, (s1, s2, w1,
               w2) in enumerate(tqdm(zip(shares1, shares2, weights1,
                                         weights2))):
        # We assume 1-indexed districts.
        for dist, v in s1.items():
            state1[dist] = v
        for dist, v in s2.items():
            state2[dist] = v
        for k, v in enumerate(sorted(state1)):
            hist1[k][v] += w1
        for k, v in enumerate(sorted(state2)):
            hist2[k][v] += w2
        if step > 0 and step % resolution == 0:
            distance = 0
            for dist1, dist2 in zip(hist1, hist2):
                distance += wasserstein_distance(list(dist1.keys()),
                                                 list(dist2.keys()),
                                                 list(dist1.values()),
                                                 list(dist2.values()))
            xticks.append(step)
            trace.append(distance)
    return xticks, trace


def raw_to_hists(shares, weights):
    n_districts = len(shares[0])
    state = np.zeros(n_districts)
    hists = [Counter() for _ in range(n_districts)]
    for step, weight in zip(shares, weights):
        # We assume 1-indexed districts.
        for dist, v in step.items():
            state[int(dist) - 1] = v
        for k, v in enumerate(sorted(state)):
            hists[k][v] += weight
    return hists


def summary_stats(hist: Dict[float, float]) -> Dict[str, float]:
    items = sorted(hist.items(), key=lambda kv: kv[0])
    vals = np.array([kv[0] for kv in items])
    weights = np.array([kv[1] for kv in items])
    size = len(items)
    # see https://stackoverflow.com/a/22639392
    percentiles = 100 * np.cumsum(weights) / np.sum(weights)

    def percentile(p: float) -> float:
        idx = max(min(len(percentiles[percentiles <= p]) - 1, size - 1), 0)
        return vals[idx]

    # TODO (mean, median, q1, q3, configurable tails, mean, mode,
    #       stddev, min, max)
    mean = np.average(vals, weights=weights)
    # NumPy does not include a weighted stddev function. See
    # https://stackoverflow.com/a/2415343
    stddev = np.sqrt(np.average((vals - mean)**2, weights=weights))

    # TODO: what percentiles should be included here?
    # (e.g. 68-95-99.7?) Can we compute them more efficiently?
    return {
        'mean': mean,
        'stddev': stddev,
        'p0.1': percentile(0.1),
        'p1': percentile(1),
        'p5': percentile(5),
        'p10': percentile(10),
        'q1': percentile(25),
        'median': percentile(50),
        'q3': percentile(75),
        'p90': percentile(90),
        'p95': percentile(95),
        'p99': percentile(99),
        'p99.9': percentile(99.9),
        'min': np.min(vals),
        'max': np.max(vals),
        'modes': list(vals[weights == np.max(weights)])
    }


def box_style(bp, color, ax, label=None):
    # from https://stackoverflow.com/a/20132614
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['fliers'], markerfacecolor=color, marker='.', markersize=7)
    if label:
        ax.plot([], c=color, label=label)


def shares_boxplot_by_chain(chains: List,
                            labels: List[str],
                            ax,
                            whislo_col: str = 'p1',
                            whishi_col: str = 'p99',
                            min_col: Optional[str] = 'min',
                            max_col: Optional[str] = 'max',
                            mean_col: Optional[str] = None,
                            cmap: str = 'tab20',
                            colors: Optional[List[str]] = None,
                            **kwargs):
    n_chains = len(chains)
    n_districts = len(chains[0])

    def color(idx):
        if colors:
            return colors[idx % len(colors)]
        return get_cmap(cmap)(idx)

    offset = n_chains / 2
    for idx, (chain, label) in enumerate(zip(chains, labels)):
        summary = [summary_stats(dist) for dist in chain]
        boxes = []
        for district in summary:
            bxp_data = {
                'q1': district['q1'],
                'med': district['median'],
                'q3': district['q3'],
                'fliers': []
            }
            if whislo_col and whishi_col:
                bxp_data['whislo'] = district[whislo_col]
                bxp_data['whishi'] = district[whishi_col]
            if mean_col:
                bxp_data['mean'] = district[mean_col]
            if min_col and max_col:
                bxp_data['fliers'] = [district[min_col], district[max_col]]
            boxes.append(bxp_data)
        bp = ax.bxp(boxes,
                    showmeans=(mean_col is not None),
                    positions=(n_chains * np.arange(n_districts)) +
                    (0.8 * (idx - offset)))
        box_style(bp, color(idx), ax, label=label)
    ax.set_xticks(range(0, n_chains * n_districts, n_chains))
    ax.set_xticklabels(range(1, n_districts + 1))


def load_from_jsonl(filename, end_step=None):
    """Loads spanning tree log-counts from a JSONL file."""
    log_counts = []
    weights = []
    with open(filename) as f:
        meta = json.loads(f.readline())['meta']
        init = json.loads(f.readline())['init']
        st_init = np.array(init['spanning_tree_counts'])
        log_counts.append({
            dist: float(np.log(s))
            for dist, s in enumerate(st_init)
        })
        for idx, line in enumerate(tqdm(f)):
            step = json.loads(line)['step']
            if end_step is not None and idx > end_step:
                break
            weight = 1 + sum(
                step['counts'].get(c, 0)
                for c in ('no_split', 'non_adjacent', 'seam_length'))
            log_counts.append({
                dist: float(np.log(c))
                for dist, c in zip(step['dists'], step['spanning_tree_counts'])
            })
            weights.append(weight)
    # correct weight lag by removing last step
    return log_counts[:-1], weights, meta


if __name__ == '__main__':
    main()
