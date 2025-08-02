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

def get_dir(main_dir: str, dataset: str, net: str):
    return os.path.join(main_dir, 'report', dataset, net)

def accumulate_sparsity_report(input_dir: str, dataset: str, sparsity_type: str):
    sparsity_type_display = str(sparsity_type).split('.')[1]
    # print(f'Accumulating Sparsity Report for {dataset} - {sparsity_type_display}')

    bp = [[], [], []]
    ff = [[], []]
    ffrnn = [[], []]
    
    dir = get_dir(input_dir, dataset, 'bp')
    files = [os.path.join(dir, file) for file in sorted(os.listdir(dir)) if file.endswith('.json')]
    args_list = [(path, sparsity_type) for path in files]
    with Pool() as pool:
        results = pool.map(process_bp_file, args_list)
    for i in range(3):
        bp[i].extend([res[i] for res in results])

    dir = get_dir(input_dir, dataset, 'ff')
    files = [os.path.join(dir, file) for file in sorted(os.listdir(dir)) if file.endswith('.json')]
    args_list = [(path, sparsity_type) for path in files]
    with Pool() as pool:
        results = pool.map(process_ff_file, args_list)
    for i in range(2):
        ff[i].extend([res[i] for res in results])

    if FFRNN:
        dir = get_dir(input_dir, dataset, 'ffrnn')
        files = [os.path.join(dir, file) for file in sorted(os.listdir(dir)) if file.endswith('.json')]
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

def plot_accumulated_sparsity_report_for_all_datasets(input_dir : str, sparsity_type, min_y=None, max_y=None, save=False, output=None):
    datasets = ['mnist', 'fashion', 'cifar10']
    fig, axes = plt.subplots(1, 3, figsize=(12, 12/3), sharey=True)
    for i, dataset in enumerate(datasets):
        bp, ff, ffrnn = accumulate_sparsity_report(input_dir, dataset, sparsity_type)
        
        for j in range(3):
            if len(bp[j]) == 0:
                continue
            mean = np.nanmean(bp[j], axis=0)
            std = np.nanstd(bp[j], axis=0)
            x = np.arange(len(mean))
            axes[i].plot(x, mean, label=f'BPNN Layer {j+1}')
            axes[i].fill_between(x, mean-std, mean+std, alpha=0.2)

        for j in range(2):
            if len(ff[j]) == 0:
                continue
            mean = np.nanmean(ff[j], axis=0)
            std = np.nanstd(ff[j], axis=0)
            x = np.arange(len(mean))
            axes[i].plot(x, mean, label=f'FFNN Layer {j+1}')
            axes[i].fill_between(x, mean-std, mean+std, alpha=0.2)

        if FFRNN:
            for j in range(2):
                if len(ffrnn[j]) == 0:
                    continue
                mean = np.nanmean(ffrnn[j], axis=0)
                std = np.nanstd(ffrnn[j], axis=0)
                x = np.arange(len(mean))
                axes[i].plot(x, mean, label=f'FFRNN Layer {j+1}')
                axes[i].fill_between(x, mean-std, mean+std, alpha=0.2)


        axes[i].set_xlabel('Training Batch')
        axes[i].set_ylabel('Sparsity')
        axes[i].grid(True)

    # Set ticks from arguments
    if min_y is not None and max_y is not None:
        for ax in axes:
            ax.set_ylim(min_y, max_y)

    # Create a legend in the second row
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.95))
    fig.subplots_adjust(
        left=0.065,
        bottom=0.2,
        right=0.968,
        top=0.825,
        wspace=0.122,
        hspace=0.198
    )

    if save:
        if output is None:
            output = f'sparsity_report_{str(sparsity_type).split(".")[1]}.png'
        plt.savefig(output, bbox_inches='tight')
    else:
        plt.show()

def plot_accumulated_sparsity_report_for_all_datasets_neurons(
    input_dir : str,
    sparsity_type,
    min_y=None,
    max_y=None,
    save=False,
    output=None
):
    datasets = ['mnist', 'fashion', 'cifar10']
    fig, axes = plt.subplots(1, 3, figsize=(12, 12/3), sharey=True)

    for i, dataset in enumerate(datasets):
        bp, ff, ffrnn = accumulate_sparsity_report(input_dir, dataset, sparsity_type)
        
        if all(len(bp[j]) > 0 for j in range(3)):
            bp_stack = np.stack([np.array(bp[j]) for j in range(3)], axis=0)  # (3, num_runs, num_batches)
            dir = get_dir(input_dir, dataset, 'bp')
            files = [file.replace('.json', '').split('_')[0] for file in sorted(os.listdir(dir)) if file.endswith('.json')]

            # Average across layers, keep runs separate
            avg_across_layers = np.nanmean(bp_stack, axis=0)  # (num_runs, num_batches)
            for run_idx, run in enumerate(avg_across_layers):
                x = np.arange(len(run))
                axes[i].plot(x, run, label=f'BPNN, N = {files[run_idx]}' if i == 0 else None)

        if all(len(ff[j]) > 0 for j in range(2)):
            ff_stack = np.stack([np.array(ff[j]) for j in range(2)], axis=0)
            dir = get_dir(input_dir, dataset, 'ff')
            files = [file.replace('.json', '').split('_')[0] for file in sorted(os.listdir(dir)) if file.endswith('.json')]

            avg_across_layers = np.nanmean(ff_stack, axis=0)
            for run_idx, run in enumerate(avg_across_layers):
                x = np.arange(len(run))
                axes[i].plot(x, run, label=f'FFNN, N = {files[run_idx]}' if i == 0 else None)

        if FFRNN and all(len(ffrnn[j]) > 0 for j in range(2)):
            ffrnn_stack = np.stack([np.array(ffrnn[j]) for j in range(2)], axis=0)
            dir = get_dir(input_dir, dataset, 'ffrnn')
            files = [file.replace('.json', '').split('_')[0] for file in sorted(os.listdir(dir)) if file.endswith('.json')]

            avg_across_layers = np.nanmean(ffrnn_stack, axis=0)
            for run_idx, run in enumerate(avg_across_layers):
                x = np.arange(len(run))
                axes[i].plot(x, run, label=f'FFRNN {files[run_idx]}' if i == 0 else None)

        axes[i].set_xlabel('Training Batch')
        axes[i].set_ylabel('Sparsity')
        axes[i].grid(True)

    # Set ticks from arguments
    if min_y is not None and max_y is not None:
        for ax in axes:
            ax.set_ylim(min_y, max_y)

    # Legend stuff
    handles, labels = axes[0].get_legend_handles_labels()

    # Separate handles/labels for BPNN and FFNN
    bpnn_handles = []
    bpnn_labels = []
    ffnn_handles = []
    ffnn_labels = []
    for h, l in zip(handles, labels):
        if l and l.startswith('BPNN'):
            bpnn_handles.append(h)
            bpnn_labels.append(l)
        elif l and l.startswith('FFNN'):
            ffnn_handles.append(h)
            ffnn_labels.append(l)

    def extract_n(label):
        try:
            return int(label.split('=')[-1].strip())
        except Exception:
            return 0

    bpnn_sorted = sorted(zip(bpnn_labels, bpnn_handles), key=lambda x: extract_n(x[0]))
    ffnn_sorted = sorted(zip(ffnn_labels, ffnn_handles), key=lambda x: extract_n(x[0]))
    bpnn_labels, bpnn_handles = zip(*bpnn_sorted) if bpnn_sorted else ([], [])
    ffnn_labels, ffnn_handles = zip(*ffnn_sorted) if ffnn_sorted else ([], [])

    # Create two legends
    print("BPNN Legend labels: ", bpnn_labels)
    print("FFNN Legend labels: ", ffnn_labels)
    legend1 = fig.legend(bpnn_handles, bpnn_labels, loc='upper center', ncol=len(bpnn_labels), bbox_to_anchor=(0.5, 1.02), frameon=False)
    legend2 = fig.legend(ffnn_handles, ffnn_labels, loc='upper center', ncol=len(ffnn_labels), bbox_to_anchor=(0.5, 0.96), frameon=False)
    
    # Optionally add FFRNN if enabled and present
    if FFRNN:
        ffrnn_handles = [h for h, l in zip(handles, labels) if l and l.startswith('FFRNN')]
        ffrnn_labels = [l for l in labels if l and l.startswith('FFRNN')]
        if ffrnn_handles:
            fig.legend(ffrnn_handles, ffrnn_labels, loc='upper center', ncol=len(ffrnn_labels), bbox_to_anchor=(0.5, 0.96), frameon=False)
    
    fig.subplots_adjust(
        left=0.065,
        bottom=0.2,
        right=0.968,
        top=0.835,
        wspace=0.122,
        hspace=0.198
    )

    if save:
        if output is None:
            output = f'sparsity_report_{str(sparsity_type).split(".")[1]}.png'
        plt.savefig(output, bbox_inches='tight', dpi=600)
    else:
        plt.show()

# %% Read sparsity report and plot
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot accumulated sparsity report. We use multithreading to speed up the loading of the sparsity reports.')
    parser.add_argument('-i', '--input-dir', type=str, required=True, help='Input Directory')
    parser.add_argument('-t', '--sparsity-type', type=str, required=True, help='Type of sparsity to plot')
    parser.add_argument('--enable-ffrnn', action='store_true', help='Enable FFRNN sparsity report')
    parser.add_argument('--min-y', type=float, default=None, help='Minimum y-axis value')
    parser.add_argument('--max-y', type=float, default=None, help='Maximum y-axis value')
    parser.add_argument('--save', action='store_true', help='Save the plot instead of showing it')
    parser.add_argument('--output', type=str, default=None, help='Output file to save the plot')
    parser.add_argument('--plot-type', type=str, choices=['layers', 'neurons'], default='neurons',
                        help='Type of plot: "layers" for individual layers, "neurons" for multiple layer sizes (default: neurons)')
    args = parser.parse_args()

    FFRNN = args.enable_ffrnn

    if args.plot_type == 'layers':
        plot_accumulated_sparsity_report_for_all_datasets(
            input_dir=args.input_dir,
            sparsity_type=args.sparsity_type,
            min_y=args.min_y,
            max_y=args.max_y,
            save=args.save,
            output=args.output
        )
    else:
        plot_accumulated_sparsity_report_for_all_datasets_neurons(
            input_dir=args.input_dir,
            sparsity_type=args.sparsity_type,
            min_y=args.min_y,
            max_y=args.max_y,
            save=args.save,
            output=args.output
        )
    