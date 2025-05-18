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
display_names = ['BP', 'FFNN', 'FFNN+C', 'FFRNN']

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
    # prune_reports_directory = 'prune_reports_max_last_10t'
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

        plt.title('Accuracy vs. Compression ratio')
        plt.xscale('log', base=2)
        # plt.gca().invert_xaxis()
        plt.xlabel('Compression ratio')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.xticks([1, 2, 8, 32, 128, 512, 2048])
        # plt.yticks([0, 10, 20, 30, 40, 50, 60])
        # plt.yticks([0, 20, 40, 60, 80, 100])

        xticks = [f'{tick}' for tick in plt.gca().get_xticks()]
        plt.gca().set_xticklabels(xticks)
        
        # yticks = [f'{tick}%' for tick in plt.gca().get_yticks()]
        # plt.gca().set_yticklabels(yticks)

        plt.show()

    _plot_pruning_ratios()

plot_pruning_ratios('prune_reports_fashion_10t')

# %% Plot Sparsity

def get_sparsity(network_type: str, report: Dict):
    result = []
    for key in report.keys():
        hoyer = report[key]['HOYER']
        if network_type == 'ff':
            layer1 = hoyer['layer_0']
            layer2 = hoyer['layer_1']
        if network_type == 'ffc':
            layer1 = hoyer['layer_0']
            layer2 = hoyer['layer_1']
        if network_type == 'ffrnn':
            layer1 = hoyer['layer_1']['fw+bw']
            layer2 = hoyer['layer_2']['fw+bw']
        if network_type == 'bp':
            layer1 = hoyer['layer0']['weight']
            layer2 = hoyer['layer1']['weight']
        result.append((layer1, layer2))
    return result

# def plot_sparsity0():
#     sparsities = [get_sparsity(network_types[i], reports[i]) for i in range(len(reports))]
#     for i in range(len(sparsities)):
#         if display_names[i] == 'BP':
#             continue
#         l1, l2 = zip(*sparsities[i])
#         plt.plot(l1, color=colors[i], label=display_names[i])

#     plt.legend()
#     plt.show()

# %%
def read_sparsity_report(network_type: str, sparsity_type: SparsityType):
    with open(f"./sparsity_report_fashion/{network_type}.json", "r") as f:
        data = f.read()
        dict = json.loads(data)

        result = { }
        for type in SparsityType:
            result[str(type).split('.')[1]] = []
        
        for epoch in dict.keys():
            for batch in dict[epoch]:
                for type in SparsityType:
                    t = str(type).split('.')[1]
                    result[t].append(batch[t])
        
        t = str(sparsity_type).split('.')[1]
        return result[t]

# %%

plt.rcParams.update({'font.size': 6})

def plot_sparsity_multiple_networks(axes, sparsity_type: SparsityType, sparsity_name: str):
    bp_sparsity = read_sparsity_report('bp', sparsity_type)
    ff_sparsity = read_sparsity_report('ff', sparsity_type)
    ffrnn_sparsity = read_sparsity_report('ffrnn', sparsity_type)

    max_batch = len(bp_sparsity)
    bp1 = [x['layer_0']['weight'] for x in bp_sparsity[:max_batch]]
    bp2 = [x['layer_1']['weight'] for x in bp_sparsity[:max_batch]]
    bp3 = [x['layer_2']['weight'] for x in bp_sparsity[:max_batch]]

    ff1 = [x['layer_0'] for x in ff_sparsity[:max_batch]]
    ff2 = [x['layer_1'] for x in ff_sparsity[:max_batch]]

    ffrnn1 = [x['layer_1']['fw+bw'] for x in ffrnn_sparsity[:max_batch]]
    ffrnn2 = [x['layer_2']['fw+bw'] for x in ffrnn_sparsity[:max_batch]]

    LW = 0.8
    axes.plot(bp1, color='orange', label='BP Layer 1', linewidth=LW)
    axes.plot(bp2, color='red', label='BP Layer 2', linewidth=LW)
    axes.plot(bp3, color='purple', label='BP Layer 3', linewidth=LW)

    axes.plot(ff1, color='darkblue', label='FFNN Layer 1', linewidth=LW)
    axes.plot(ff2, color='blue', label='FFNN Layer 2', linewidth=LW)

    # Too cluttered...
    # axes.plot(ffrnn1, color='darkgreen', label='FFRNN Layer 1', linewidth=LW)
    # axes.plot(ffrnn2, color='green', label='FFRNN Layer 2', linewidth=LW)

    axes.set_xlabel('Training Batch')
    axes.set_ylabel('Sparsity')
    axes.set_title(f'{sparsity_name} over Training Batch')
    axes.grid()

    axes.legend(loc='upper right', fontsize=4)

fig, axes = plt.subplots(2, 2, figsize=(6, 5), dpi=300)
plot_sparsity_multiple_networks(axes[0][0], SparsityType.L1_NEG_ENTROPY, "L1-normalized Negative Entropy")
plot_sparsity_multiple_networks(axes[0][1], SparsityType.L2_NEG_ENTROPY, "L2-normalized Negative Entropy")
plot_sparsity_multiple_networks(axes[1][0], SparsityType.HOYER, "Hoyer Sparsity")
plot_sparsity_multiple_networks(axes[1][1], SparsityType.GINI, "Gini Sparsity")
plt.tight_layout()
plt.show()
matplotlib.style.use('default')

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
# %%
