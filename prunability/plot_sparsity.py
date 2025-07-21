# %% Imports and util functions

import os
import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import Pool
from utils_sparsity import read_sparsity_report

def process_bp_file(args):
    path, sparsity_type = args
    bp_sparsity = read_sparsity_report(path, sparsity_type)
    max_batch = len(bp_sparsity)
    # print(f'Loading BPNN Sparsity Stats... Batches: {max_batch}')
    return [[x[f'layer_{i}']['weight'] for x in bp_sparsity[:max_batch]] for i in range(3)]

def process_ff_file(args):
    path, sparsity_type = args
    ff_sparsity = read_sparsity_report(path, sparsity_type)
    max_batch = len(ff_sparsity)
    # print(f'Loading FFNN Sparsity Stats... Batches: {max_batch}')
    return [[x[f'layer_{i}'] for x in ff_sparsity[:max_batch]] for i in range(2)]

def process_ffrnn_file(args):
    path, sparsity_type = args
    ffrnn_sparsity = read_sparsity_report(path, sparsity_type)
    max_batch = len(ffrnn_sparsity)
    # print(f'Loading FFRNN Sparsity Stats... Batches: {max_batch}')
    return [[x[f'layer_{i+1}']['fw+bw'] for x in ffrnn_sparsity[:max_batch]] for i in range(2)]

NETWORKS = ['bp', 'ff', 'ffc', 'ffrnn']
DISPLAY_NAMES = ['BPNN', 'FFNN', 'FFNN+C', 'FFRNN']
FFRNN = False

def get_dir(dataset, net):
    return f"./sparsity/report/{dataset}/{net}"

def accumulate_sparsity_report(dataset, sparsity_type):
    sparsity_type_display = str(sparsity_type).split('.')[1]
    # print(f'Accumulating Sparsity Report for {dataset} - {sparsity_type_display}')

    bp = [[], [], []]
    ff = [[], []]
    ffrnn = [[], []]
    
    dir = get_dir(dataset, 'bp')
    files = [os.path.join(dir, file) for file in os.listdir(dir)]
    args_list = [(path, sparsity_type) for path in files]
    with Pool() as pool:
        results = pool.map(process_bp_file, args_list)
    for i in range(3):
        bp[i].extend([res[i] for res in results])

    dir = get_dir(dataset, 'ff')
    files = [os.path.join(dir, file) for file in os.listdir(dir)]
    args_list = [(path, sparsity_type) for path in files]
    with Pool() as pool:
        results = pool.map(process_ff_file, args_list)
    for i in range(2):
        ff[i].extend([res[i] for res in results])

    if FFRNN:
        dir = get_dir(dataset, 'ffrnn')
        files = [os.path.join(dir, file) for file in os.listdir(dir)]
        args_list = [(path, sparsity_type) for path in files]
        with Pool() as pool:
            results = pool.map(process_ffrnn_file, args_list)
        for i in range(2):
            ffrnn[i].extend([res[i] for res in results])

    return bp, ff, ffrnn

def plot_accumulated_sparsity_report(dataset, sparsity_type, bp, ff, ffrnn, min_y, max_y):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(3):
        if len(bp[i]) == 0:
            continue
        mean = np.mean(bp[i], axis=0)
        std = np.std(bp[i], axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=f'BPNN Layer {i+1}')
        ax.fill_between(x, mean-std, mean+std, alpha=0.2)

    for i in range(2):
        if len(ff[i]) == 0:
            continue
        mean = np.mean(ff[i], axis=0)
        std = np.std(ff[i], axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=f'FFNN Layer {i+1}')
        ax.fill_between(x, mean-std, mean+std, alpha=0.2)

    if FFRNN:
        for i in range(2):
            if len(ffrnn[i]) == 0:
                continue
            mean = np.mean(ffrnn[i], axis=0)
            std = np.std(ffrnn[i], axis=0)
            x = np.arange(len(mean))
            ax.plot(x, mean, label=f'FFRNN Layer {i+1}')
            ax.fill_between(x, mean-std, mean+std, alpha=0.2)
    
    # Set ticks from arguments
    if min_y is not None and max_y is not None:
        ax.set_ylim(min_y, max_y)

    ax.set_title(f'Sparsity Report for {dataset} - {str(sparsity_type).split(".")[1]}')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Sparsity')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# %% Read sparsity report and plot
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot accumulated sparsity report. We use multithreading to speed up the loading of the sparsity reports.')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('-t', '--sparsity_type', type=str, required=True, help='Type of sparsity to plot')
    parser.add_argument('--enable-ffrnn', action='store_true', help='Enable FFRNN sparsity report')
    parser.add_argument('--min-y', type=float, default=None, help='Minimum y-axis value')
    parser.add_argument('--max-y', type=float, default=None, help='Maximum y-axis value')
    args = parser.parse_args()

    FFRNN = args.enable_ffrnn

    bp, ff, ffrnn = accumulate_sparsity_report(args.dataset, args.sparsity_type)
    plot_accumulated_sparsity_report(args.dataset, args.sparsity_type, bp, ff, ffrnn, args.min_y, args.max_y)
