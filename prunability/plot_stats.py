# %% Imports
import torch
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from fflib.enums import SparsityType

from scipy.stats import ttest_ind
from typing import Dict

display_names_dict = {
    'bp': 'BPNN',
    'ff': 'FFNN2',
    'ff_v3': 'FFNN3',
    'ffc': 'FFNN+C',
    'ffrnn': 'FFRNN'
}

# %% Min Max Statistics
def compute_iqr(tensor):
    q75, q25 = torch.quantile(tensor, 0.75), torch.quantile(tensor, 0.25)
    return (q75 - q25).item()

def format_float(x):
    return f"{x:.2f}"

def compare(network_type: str):
    with open(f"./minimize_reports/{network_type}.json", "r") as f:
        data = json.load(f)

    mx = data['mx']
    mn = data['mn']

    mx_pretest = torch.tensor([x['pretest'] for x in mx]) * 100
    mn_pretest = torch.tensor([x['pretest'] for x in mn]) * 100

    # Compute statistics
    stats = {
        'name': network_type,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        'mean_mx': mx_pretest.mean().item(),
        'mean_mn': mn_pretest.mean().item(),
        'std_mx': mx_pretest.std().item(),
        'std_mn': mn_pretest.std().item(),
        'max_mx': mx_pretest.max().item(),
        'max_mn': mn_pretest.max().item(),
        'iqr_mx': compute_iqr(mx_pretest),
        'iqr_mn': compute_iqr(mn_pretest),
        'pvalue': ttest_ind(mx_pretest, mn_pretest, equal_var=False).pvalue
    }

    return stats

# List of all networks to include
network_names = ["ff", "ffc", "ffrnn", "ff_v3"]
results = [compare(name) for name in network_names]

# Print LaTeX table
print(r"\begin{tabular}{lccccccc}")
print(r"\textbf{Network} & Mean$_{mx}$ & Mean$_{mn}$ & Std$_{mx}$ & Std$_{mn}$ & Max$_{mx}$ & Max$_{mn}$ & p-value \\ \hline")
for r in results:
    print(f"{display_names_dict[r['name']]} & {format_float(r['mean_mx'])} & {format_float(r['mean_mn'])} & "
          f"{format_float(r['std_mx'])} & {format_float(r['std_mn'])} & "
          f"{format_float(r['max_mx'])} & {format_float(r['max_mn'])} & {r['pvalue']:.2e} \\\\")
print(r"\end{tabular}")

# %% Make table of stats

import json
import torch

table_rows = []

for network_type in ['ff', 'ff_v3', 'ffc', 'ffrnn']:
    with open(f"./minimize_reports2/{network_type}.json", "r") as f:
        data = json.load(f)

    mx = data['mx']
    mn = data['mn']
    
    weight = 'fw' if network_type == 'ffrnn' else 'weight'
    
    mx_min_layer_0 = torch.tensor([x['net_stats']['layer_0'][weight]['min'] for x in mx]).mean().item()
    mx_max_layer_0 = torch.tensor([x['net_stats']['layer_0'][weight]['max'] for x in mx]).mean().item()
    mx_mean_layer_0 = torch.tensor([x['net_stats']['layer_0'][weight]['mean'] for x in mx]).mean().item()
    mx_std_layer_0 = torch.tensor([x['net_stats']['layer_0'][weight]['std'] for x in mx]).mean().item()
    
    mx_min_layer_1 = torch.tensor([x['net_stats']['layer_1'][weight]['min'] for x in mx]).mean().item()
    mx_max_layer_1 = torch.tensor([x['net_stats']['layer_1'][weight]['max'] for x in mx]).mean().item()
    mx_mean_layer_1 = torch.tensor([x['net_stats']['layer_1'][weight]['mean'] for x in mx]).mean().item()
    mx_std_layer_1 = torch.tensor([x['net_stats']['layer_1'][weight]['std'] for x in mx]).mean().item()

    if network_type == 'ff_v3':
        mx_min_layer_2 = torch.tensor([x['net_stats']['layer_2'][weight]['min'] for x in mx]).mean().item()
        mx_max_layer_2 = torch.tensor([x['net_stats']['layer_2'][weight]['max'] for x in mx]).mean().item()
        mx_mean_layer_2 = torch.tensor([x['net_stats']['layer_2'][weight]['mean'] for x in mx]).mean().item()
        mx_std_layer_2 = torch.tensor([x['net_stats']['layer_2'][weight]['std'] for x in mx]).mean().item()

        mn_min_layer_2 = torch.tensor([x['net_stats']['layer_2'][weight]['min'] for x in mn]).mean().item()
        mn_max_layer_2 = torch.tensor([x['net_stats']['layer_2'][weight]['max'] for x in mn]).mean().item()
        mn_mean_layer_2 = torch.tensor([x['net_stats']['layer_2'][weight]['mean'] for x in mn]).mean().item()
        mn_std_layer_2 = torch.tensor([x['net_stats']['layer_2'][weight]['std'] for x in mn]).mean().item()

    mn_min_layer_0 = torch.tensor([x['net_stats']['layer_0'][weight]['min'] for x in mn]).mean().item()
    mn_max_layer_0 = torch.tensor([x['net_stats']['layer_0'][weight]['max'] for x in mn]).mean().item()
    mn_mean_layer_0 = torch.tensor([x['net_stats']['layer_0'][weight]['mean'] for x in mn]).mean().item()
    mn_std_layer_0 = torch.tensor([x['net_stats']['layer_0'][weight]['std'] for x in mn]).mean().item()
    
    mn_min_layer_1 = torch.tensor([x['net_stats']['layer_1'][weight]['min'] for x in mn]).mean().item()
    mn_max_layer_1 = torch.tensor([x['net_stats']['layer_1'][weight]['max'] for x in mn]).mean().item()
    mn_mean_layer_1 = torch.tensor([x['net_stats']['layer_1'][weight]['mean'] for x in mn]).mean().item()
    mn_std_layer_1 = torch.tensor([x['net_stats']['layer_1'][weight]['std'] for x in mn]).mean().item()
    
    table_rows.append(f"\\multirow{{4}}{{*}}{{{display_names_dict[network_type]}}} & \\multirow{{2}}{{*}}{{mx}} & Layer 1 & {mx_min_layer_0:.2f} & {mx_max_layer_0:.2f} & {mx_mean_layer_0:.2f} & {mx_std_layer_0:.2f} \\\\")
    table_rows.append(f"                                       &                       & Layer 2 & {mx_min_layer_1:.2f} & {mx_max_layer_1:.2f} & {mx_mean_layer_1:.2f} & {mx_std_layer_1:.2f} \\\\")
    
    if network_type == 'ff_v3':
        table_rows.append(f"                                       &                       & Layer 3 & {mx_min_layer_2:.2f} & {mx_max_layer_2:.2f} & {mx_mean_layer_2:.2f} & {mx_std_layer_2:.2f} \\\\")

    table_rows.append(f"                                       & \\multirow{{2}}{{*}}{{mn}} & Layer 1 & {mn_min_layer_0:.2f} & {mn_max_layer_0:.2f} & {mn_mean_layer_0:.2f} & {mn_std_layer_0:.2f} \\\\")
    table_rows.append(f"                                       &                       & Layer 2 & {mn_min_layer_1:.2f} & {mn_max_layer_1:.2f} & {mn_mean_layer_1:.2f} & {mn_std_layer_1:.2f}")

    if network_type == 'ff_v3':
        table_rows[-1] += f"\\\\"
        table_rows.append(f"                                       &                       & Layer 3 & {mn_min_layer_2:.2f} & {mn_max_layer_2:.2f} & {mn_mean_layer_2:.2f} & {mn_std_layer_2:.2f}")

    table_rows[-1] += f"\\vspace{{5px}}\\\\"

# LaTeX table header
latex_table = """
\\begin{table}[ht]
\\centering
\\begin{tabular}{lcccccc}
\\textbf{Network Type} & Opt & \\textbf{Layer} & \\textbf{Min} & \\textbf{Max} & \\textbf{Mean} & \\textbf{Std} \\\\
\\hline
"""

# Add rows to the LaTeX table
latex_table += "\n".join(table_rows)

# LaTeX table footer
latex_table += """
\\end{tabular}
\\caption{Statistics for network weight matrices for different network types across layers.}
\\label{table:network_stats}
\\end{table}
"""

# Output the generated LaTeX table
print(latex_table)

# %% Plot Pruning Ratios
network_types = ['bp', 'ff', 'ffc', 'ffrnn']
colors = ['red', 'green', 'blue', 'purple']
# colors = ["#ff1d1d", '#00ff00', "#00d9ff", '#ff00ff'] # dark
display_names = ['BPNN', 'FFNN', 'FFNN+C', 'FFRNN']

def plot_pruning_ratios(prune_reports_directory: str):
    print(prune_reports_directory)
    def open_prune_report(prune_reports_dir: str, network_type: str):
        with open(f"./{prune_reports_dir}/{network_type}.json", "r") as f:
            data = f.read()
            report = json.loads(data)
            return report
        
    bp = open_prune_report(prune_reports_directory, 'bp')
    ff = open_prune_report(prune_reports_directory, 'ff')
    ffc = open_prune_report(prune_reports_directory, 'ffc')
    ffrnn = open_prune_report(prune_reports_directory, 'ffrnn')
    reports = [bp, ff, ffc, ffrnn]

    def _plot_pruning_ratios():
        # acc/ratio
        def get_acc_ratio(report: Dict):
            mx_pruned_size = max(report[key]['pruned_size'] for key in report.keys())
            return [
                (int(mx_pruned_size) / report[neurons]['pruned_size'], report[neurons]['test'] * 100)
                for neurons in map(str, sorted(map(int, report.keys())))
            ]
        
        def get_acc_ratio_tries(report: Dict):
            mx_pruned_size = max(
                trial['pruned_size'] for trials in report.values() for trial in trials
            )

            acc_ratios = []
            for neurons in sorted(map(int, report.keys())):
                trials = report[str(neurons)]
                ratios = []
                accuracies = []

                for trial in trials:
                    ratio = int(mx_pruned_size) / trial['pruned_size']
                    # ratio = trial['pruned_size']
                    # ratio = float(neurons)
                    acc = trial['test'] * 100
                    ratios.append(ratio)
                    accuracies.append(acc)

                mean_ratio = torch.tensor(ratios).mean().item()
                accuracies_tensor = torch.tensor(accuracies)
                mean_acc = accuracies_tensor.mean().item()
                std_acc = accuracies_tensor.std().item()
                min_acc = accuracies_tensor.min().item()
                max_acc = accuracies_tensor.max().item()

                acc_ratios.append((mean_ratio, mean_acc, std_acc, min_acc, max_acc))

            return acc_ratios

        plt.figure(figsize=(10, 6), dpi = 300)

        # acc_ratios = [get_acc_ratio(report) for report in reports]
        # for i in range(len(acc_ratios)):
        #     x, y = zip(*acc_ratios[i])
        #     plt.plot(x, y, color=colors[i], label=display_names[i], marker='o')

        acc_ratios_tries = [get_acc_ratio_tries(report) for report in reports]

        for i, data in enumerate(acc_ratios_tries):
            x, y, yerr, ymin, ymax = zip(*data)
            x = torch.tensor(x)
            y = torch.tensor(y)
            yerr = torch.tensor(yerr)
            ymin = torch.tensor(ymin)
            ymax = torch.tensor(ymax)

            # Plot mean line
            plt.plot(x, y, color=colors[i], label=display_names[i], marker='o')

            # Plot shaded error region
            plt.fill_between(x, y - yerr, y + yerr, color=colors[i], alpha=0.2)

            # Transparent min/max lines
            plt.plot(x, ymin, linestyle='--', color=colors[i], alpha=0.1)
            plt.plot(x, ymax, linestyle='--', color=colors[i], alpha=0.1)

        # plt.title('Accuracy vs. Compression ratio')
        plt.xscale('log', base=2)
        # plt.gca().invert_xaxis()
        plt.xlabel('Compression ratio')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.xticks([1, 2, 8, 32, 128, 512, 2048])
        # plt.yticks([0, 10, 20, 30, 40, 50, 60]) # CIFAR10
        plt.yticks([0, 20, 40, 60, 80, 100]) # MNIST, FashionMNIST

        xticks = [f'{tick}' for tick in plt.gca().get_xticks()]
        plt.gca().set_xticklabels(xticks)
        
        yticks = [f'{tick}%' for tick in plt.gca().get_yticks()]
        plt.gca().set_yticklabels(yticks)

        plt.show()

    _plot_pruning_ratios()

matplotlib.style.use('default')

plt.rcParams.update({'font.size': 14})
# # Customizing the style to a light gray background
# plt.rcParams.update({
#     'axes.facecolor': "#000000",  # Dark gray background for the axes
#     'figure.facecolor': "#000000",  # Slightly lighter gray for the figure
#     'axes.edgecolor': 'white',     # White edges for contrast
#     'text.color': 'white',         # White text for readability
#     'axes.labelcolor': 'white',    # White labels
#     'xtick.color': 'white',        # White x-axis ticks
#     'ytick.color': 'white'         # White y-axis ticks
# })

plot_pruning_ratios('prune_reports_min_10t')
# plot_pruning_ratios('prune_reports_fashion_10t')
# plot_pruning_ratios('prune_reports_cifar10_10t')

# %%

colors = ['orange', 'red', 'purple', 'darkblue', 'blue']
# colors = ['#FF6700', '#FF073A', '#D100FF', "#25D3FF", "#00FFE1"]

plt.rcParams.update({'font.size': 6})

def plot_sparsity_multiple_networks(axes, sparsity_report: str, sparsity_type: SparsityType, sparsity_name: str):
    bp_sparsity = read_sparsity_report(sparsity_report, 'bp', sparsity_type)
    ff_sparsity = read_sparsity_report(sparsity_report, 'ff', sparsity_type)
    ffrnn_sparsity = read_sparsity_report(sparsity_report, 'ffrnn', sparsity_type)

    max_batch = len(bp_sparsity)
    bp1 = [x['layer_0']['weight'] for x in bp_sparsity[:max_batch]]
    bp2 = [x['layer_1']['weight'] for x in bp_sparsity[:max_batch]]
    bp3 = [x['layer_2']['weight'] for x in bp_sparsity[:max_batch]]

    ff1 = [x['layer_0'] for x in ff_sparsity[:max_batch]]
    ff2 = [x['layer_1'] for x in ff_sparsity[:max_batch]]

    ffrnn1 = [x['layer_1']['fw+bw'] for x in ffrnn_sparsity[:max_batch]]
    ffrnn2 = [x['layer_2']['fw+bw'] for x in ffrnn_sparsity[:max_batch]]

    LW = 0.8
    axes.plot(bp1, color=colors[0], label='BPNN Layer 1', linewidth=LW)
    axes.plot(bp2, color=colors[1], label='BPNN Layer 2', linewidth=LW)
    axes.plot(bp3, color=colors[2], label='BPNN Layer 3', linewidth=LW)

    axes.plot(ff1, color=colors[3], label='FFNN Layer 1', linewidth=LW)
    axes.plot(ff2, color=colors[4], label='FFNN Layer 2', linewidth=LW)

    # Too cluttered...
    # axes.plot(ffrnn1, color='darkgreen', label='FFRNN Layer 1', linewidth=LW)
    # axes.plot(ffrnn2, color='green', label='FFRNN Layer 2', linewidth=LW)

    axes.set_xlabel('Training Batch')
    axes.set_ylabel(sparsity_name)
    axes.grid()

    axes.legend(loc='upper right', fontsize=4)

# sparsity_report = 'sparsity_report'
# sparsity_report = 'sparsity_report_fashion'
sparsity_report = 'sparsity_report_cifar'

types = [SparsityType.L1_NEG_ENTROPY, SparsityType.L2_NEG_ENTROPY, SparsityType.HOYER, SparsityType.GINI]
type_display_names = ["L1-normalized Negative Entropy", "L2-normalized Negative Entropy", "Hoyer Sparsity", "Gini Sparsity"]
save_names = ['L1', 'L2', 'H', 'G']

for i in range(4):
    fig, axes = plt.subplots(1, 1, figsize=(9/4, 2), dpi=600)
    plot_sparsity_multiple_networks(axes, sparsity_report, types[i], type_display_names[i])
    filename = f'./figures/{sparsity_report}_{save_names[i]}.png'
    plt.tight_layout()
    plt.show()
    fig.savefig(filename)
    print(f"Saved figure at {filename}.")

# matplotlib.style.use('default')

# %%
def read_weights_histogram(network_type: str):
    with open(f"./sparsity_report_mnist5/{network_type}.json", "r") as f:
        data = f.read()
        dict = json.loads(data)

        result = []
        for epoch in dict.keys():
            for batch in dict[epoch]:
                result.append(batch['HISTOGRAM2']['layer_0']['weight'])
        
        return result

# %%

img = read_weights_histogram('bp')
img_tensor = torch.Tensor(img)
print(f"Image shape: {img_tensor.shape}")

# %%
a = img_tensor.T

plt.figure(figsize=(12, 5))
plt.imshow(a, aspect=30, cmap='gray')
plt.show()

# %%
plt.plot(img_tensor[5000])
plt.show()

# %% Baseline accuracy

network_types = ['bp', 'ff', 'ffc', 'ffrnn']
display_names = ['BPNN', 'FFNN', 'FFNN+C', 'FFRNN']
report_dirs = ['prune_reports_min_10t', 'prune_reports_fashion_10t', 'prune_reports_cifar10_10t']
dataset_names = ['MNIST', 'FashionMNIST', 'CIFAR10']

results = {name: [] for name in display_names}

def open_prune_report(prune_reports_dir: str, network_type: str):
    with open(f"./{prune_reports_dir}/{network_type}.json", "r") as f:
        report = json.load(f)
        return report

def get_max_acc(report: Dict):
    return max(trial['test'] * 100 for trials in report.values() for trial in trials)

for rid, dataset in enumerate(dataset_names):
    for n_idx, network in enumerate(network_types):
        report = open_prune_report(report_dirs[rid], network)
        max_acc = round(get_max_acc(report), 2)
        results[display_names[n_idx]].append(max_acc)

print("\\begin{table}[ht]")
print("\\centering")
print("\\caption{Baseline test accuracy (\%) across different networks and datasets.}")
print("\\label{tab:max_accuracy}")
print("\\begin{tabular}{l" + "c" * len(dataset_names) + "}")
print("\\toprule")
print("Network & " + " & ".join(dataset_names) + " \\\\")
print("\\midrule")

for name in display_names:
    accs = " & ".join(f"{acc:.2f}" for acc in results[name])
    print(f"{name} & {accs} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# %%

def plot_color_hue_mapping():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    num_classes = 10

    width = 1000
    height = 10
    spectrum_hue = np.linspace(0, 1, width)
    spectrum_hsv = np.stack([spectrum_hue, np.ones_like(spectrum_hue), np.ones_like(spectrum_hue)], axis=1)
    spectrum_rgb = mcolors.hsv_to_rgb(spectrum_hsv)
    spectrum_img = np.tile(spectrum_rgb[np.newaxis, :, :], (height, 1, 1))

    fig, ax = plt.subplots(figsize=(12, 1), dpi=100)
    ax.imshow(spectrum_img, aspect='auto')

    step = width // num_classes
    tick_positions = [i * step + step // 2 for i in range(num_classes)]
    tick_labels = [f'{i}' for i in range(num_classes)]

    for pos in tick_positions:
        ax.axvline(pos, color='white', linestyle='--', linewidth=0.5)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks([])

    ax.set_title('Color Hue Mapping')

    plt.tight_layout()
    plt.show()

# %%
