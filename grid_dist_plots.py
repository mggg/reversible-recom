"""Wasserstein trace plots for RevReCom paper (7x7 grid cut edges)."""
import glob
import json
import numpy as np
import click
import pyreadr
import geopandas as gpd

import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from gerrychain import Graph, Partition
from gerrychain.updaters import cut_edges

# weighted 7x7 cut edges distribution
GROUND_TRUTH = """32      7.32191421608e11
40      2.82316256e8
39      1.6843817279999998e9
35      1.86993438208e11
34      3.6642892581799994e11
42      1.9495219999999998e6
29      2.5066583999999997e11
37      2.6977180927999996e10
28      7.344675e10
38      7.588753206e9
31      6.983941968e11
36      7.854102064799998e10
41      3.1884255999999996e7
33      5.77889926624e11
30      5.286711393e11"""

"""
COLORS = (
    #(156/255, 180/255, 79/255),
    (149/255, 180/255, 55/255),
    (1.0, 0.75, 0.0),   # amber
    (1.0, 0.72, 0.77),  # cherryblossompink
    (1.0, 0.66, 0.07),  # darktangerine
    (0.08, 0.38, 0.74), # denim
    (0.44, 0.5, 0.56),  # slategray
    (0.82, 0.1, 0.26),  # alizarin
)
"""

COLORS = (
    (0.0, 0.5, 0.5),     # teal
    (0.82, 0.1, 0.26),   # alizarin
    (0.44, 0.5, 0.56),   # slategray
    (1.0, 0.75, 0.0),    # amber
    (0.13, 0.55, 0.13),  # forestgreen
)

"""
COLORS = [
    (0.55, 0.82, 0.77),  # lightblue
    (0.41, 0.21, 0.61),  # purpleheart
    (0.79, 0.17, 0.57),  # royalfuchsia
]
"""


LINE_STYLES = ("dashdot", "dashed", "dotted", "solid")

color_idx = 0

def next_color() -> tuple[float, float, float]:
    global color_idx
    color = COLORS[color_idx % len(COLORS)]
    color_idx += 1
    return color


@click.command()
@click.option('--chain-data', multiple=True)
@click.option('--chain-label', multiple=True)
@click.option('--fig-file', type=click.Path(), required=True)
@click.option('--smc-shapefile', type=click.Path())
@click.option('--smc-plans-file', type=click.Path(), multiple=True)
@click.option('--smc-weights-file', type=click.Path(), multiple=True)
@click.option('--smc-trace-prefix', type=click.Path())
@click.option('--wasserstein-resolution', default=10000, type=int)
def main(
    chain_data,
    chain_label,
    fig_file,
    smc_shapefile,
    smc_plans_file,
    smc_weights_file,
    smc_trace_prefix,
    wasserstein_resolution,
):
    ref_counts = [int(line.split()[0]) for line in GROUND_TRUTH.split('\n')]
    ref_weights = [float(line.split()[1]) for line in GROUND_TRUTH.split('\n')]
    cut_edge_counts = {}
    weights = {}
    for filename, label in zip(chain_data, chain_label):
        chain_cut_edge_counts, chain_weights, chain_meta = load_chain_jsonl(filename)
        print(chain_meta)
        cut_edge_counts[label] = chain_cut_edge_counts
        weights[label] = chain_weights

    fig, ax = plt.subplots(figsize=(10, 6))
    for tick in range(1, 3):
        ax.axhline(tick / 10, linewidth=0.3, color='#ccc')

    """
    for outer_idx, outer_label in enumerate(chain_label):
        for inner_idx, inner_label in enumerate(chain_label):
            if inner_idx > outer_idx:
                xs, dists = wasserstein_trace(cut_edge_counts[outer_label],
                                              cut_edge_counts[inner_label],
                                              weights[outer_label],
                                              weights[inner_label],
                                              wasserstein_resolution)
                ax.plot(
                    xs,
                    dists,
                    label=f'{outer_label} vs. {inner_label}',
                    color=next_color(),
                )
    """

    for label in chain_label:
        xs, dists = wasserstein_trace_ground_truth(
            cut_edge_counts[label],
            ref_counts,
            weights[label],
            ref_weights,
            wasserstein_resolution)
        ax.plot(xs, dists, label=f'{label} vs. full', color=next_color(), linewidth=2)

    if smc_shapefile is not None:
        gdf = gpd.read_file(smc_shapefile)
        gdf.crs = "epsg:26918"  # fake! (suppresses spurious projection warnings)
        graph = Graph.from_geodataframe(gdf)
    else:
        gdf = graph = None

    for idx, (plans_path, weights_path) in enumerate(zip(smc_plans_file, smc_weights_file)):
        smc_partitions, hist_smc_weighted = load_smc(
            graph=graph,
            rds_path=plans_path,
            weights_path=weights_path,
        )
        smc_counts = list(hist_smc_weighted.keys())
        smc_weights = list(hist_smc_weighted.values())
        smc_full_enum_distance = wasserstein_distance(
            smc_counts, ref_counts, smc_weights, ref_weights,
        )
        ax.axhline(
            smc_full_enum_distance,
            label=f"SMC (n={len(smc_partitions)}) vs. full",
            color=next_color(),
            linestyle=LINE_STYLES[idx % len(LINE_STYLES)],
        )

    if smc_trace_prefix:
        weights_files = glob.glob(f"{smc_trace_prefix}*.rds .wgt")
        trace_x = []
        trace_y = []
        for weights_path in weights_files:
            plans_path = weights_path.replace(".wgt", ".plans")
            smc_partitions, hist_smc_weighted = load_smc(
                graph=graph,
                rds_path=plans_path,
                weights_path=weights_path,
            )
            smc_counts = list(hist_smc_weighted.keys())
            smc_weights = list(hist_smc_weighted.values())
            smc_full_enum_distance = wasserstein_distance(
                smc_counts, ref_counts, smc_weights, ref_weights,
            )
            trace_x.append(int(float(plans_path.split(' __ ')[-1].split(' .rds')[0])))
            trace_y.append(smc_full_enum_distance)
            print(weights_path, "\t", smc_full_enum_distance)
        ax.scatter(trace_x, trace_y, color=next_color(), label='SMC vs. full')

    #ax.set_xlabel('Accepted steps')
    #ax.set_ylabel('Distance')
    ax.set_xlim(0, 2500000)
    ax.set_ylim(0, 1.2)
    ax.margins(0)
    ax.legend(loc="upper right", framealpha=0.65, fontsize="x-small")
    plt.savefig(fig_file, bbox_inches='tight')
    plt.close(fig)


def wasserstein_trace(counts1, counts2, weights1, weights2, resolution):
    xticks = []
    trace = []
    hist1 = Counter()
    hist2 = Counter()
    for step, (c1, c2, w1,
               w2) in enumerate(tqdm(zip(counts1, counts2, weights1,
                                         weights2))):
        # We assume 1-indexed districts.
        hist1[c1] += w1
        hist2[c2] += w2
        if step > 0 and step % resolution == 0:
            distance = wasserstein_distance(list(hist1.keys()),
                                            list(hist2.keys()),
                                            list(hist1.values()),
                                            list(hist2.values()))
            xticks.append(step)
            trace.append(distance)
    return xticks, trace


def wasserstein_trace_ground_truth(counts, ref_counts, weights, ref_weights, resolution):
    xticks = []
    trace = []
    hist = Counter()
    for step, (c, w) in enumerate(tqdm(zip(counts, weights))):
        # We assume 1-indexed districts.
        hist[c] += w
        if step > 0 and step % resolution == 0:
            distance = wasserstein_distance(
                list(hist.keys()),
                ref_counts,
                list(hist.values()),
                ref_weights
            )
            xticks.append(step)
            trace.append(distance)
    print(f"d_wass = {trace[-1]}")
    return xticks, trace


def load_chain_jsonl(filename):
    """Loads cut edge statistics from a JSONL file."""
    counts = []
    weights = []
    with open(filename) as f:
        meta = json.loads(f.readline())['meta']
        init = json.loads(f.readline())['init']
        counts.append(init['num_cut_edges'])
        for idx, line in enumerate(tqdm(f)):
            step = json.loads(line)['step']
            weight = 1 + sum(
                step['counts'].get(c, 0)
                for c in ('no_split', 'non_adjacent', 'seam_length'))
            weights.append(weight)
            counts.append(step['num_cut_edges'])

    # correct weight lag by removing last step
    return counts[:-1], weights, meta


def load_smc(graph, rds_path, weights_path=None) -> Counter:
    """Computes the cut edge count distribution of an SMC grid run."""
    run_plans = pyreadr.read_r(rds_path)
    assignments = run_plans[None].values.astype(int).T.copy()
    partitions = [
        Partition(
            assignment=dict(enumerate(row)),
            graph=graph,
            updaters={"cut_edges": cut_edges}
        )
        for row in assignments
    ]

    if weights_path is not None:
        weights = pyreadr.read_r(weights_path)[None].values.T
        weighted_hist = defaultdict(float)
        for weight, part in zip(weights[0].tolist(), partitions):
            weighted_hist[len(part["cut_edges"])] += weight
        return partitions, weighted_hist
    return partitions, Counter(len(part["cut_edges"]) for part in partitions)


if __name__ == '__main__':
    main()
