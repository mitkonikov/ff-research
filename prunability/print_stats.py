# %% Imports
import torch
import os
import asyncio
import json
import torch
import numpy as np

from fflib.enums import SparsityType
from multiprocessing import Pool
from plot_sparsity import accumulate_sparsity_report

DISPLAY_NAMES = {
    'bp': 'BPNN',
    'ff': 'FFNN',
    'ffc': 'FFNN+C',
    'ffrnn': 'FFRNN'
}

# %% Gather Statistics
def get_loader(network_type: str):
    if network_type == 'ff':
        return 'ff_net_load.py'
    if network_type == 'ff3':
        return 'ff_net_v3_load.py'
    if network_type == 'ffc':
        return 'ff_c_load.py'
    if network_type == 'ffrnn':
        return 'ff_rnn_load.py'
    if network_type == 'bp':
        return 'bp_load.py'

    raise RuntimeError("Invalid network type.")

def get_saved_location(report: str):
    with open(report, "r") as f:
        for line in f.readlines():
            if line.startswith("Model saved at"):
                model_path = line.split(" ")[-1].rstrip()
                if model_path.endswith("."):
                    model_path = model_path[:-1]
                return model_path
            
    raise RuntimeError("Invalid report. Cannot find where the model is saved...")

async def load(network: str, checkpoint: str, dataset: str, batch: int, prune_mode: str, neurons: int, seed: int, output: str | None):
    arguments = [
        'python', network,
        '-s', str(seed),
        '-i', checkpoint,
        '--pretest',
        '-d', dataset,
        '-b', str(batch),
        '-l', str(neurons),
        '--prune-mode', prune_mode
    ]

    if output is not None:
        arguments.append('--save-pruned')
        arguments.append('-o')
        arguments.append(output)
    
    proc = await asyncio.create_subprocess_exec(
        *arguments,
        stdout=asyncio.subprocess.PIPE
    )
    data = await proc.stdout.readline()
    line = data.decode('ascii').rstrip()
    await proc.wait()
    return line

def get_stats_from_model(network: str, filepath: str, dataset: str, batch: int, prune_mode: str, neurons: int, seed: int, output: str | None = None):
    model_path = filepath
    if filepath.endswith(".txt"):
        model_path = get_saved_location(filepath)
    print(f"Executing {network} {model_path} {dataset} {batch} {prune_mode} {neurons} {seed} {output}...")
    stats = asyncio.run(load(network, model_path, dataset, batch, prune_mode, neurons, seed, output))
    print(f"Done execution for {network} {model_path} {dataset} {batch} {prune_mode} {neurons} {seed} {output}...")
    stat_dict = json.loads(stats)
    stat_dict['report'] = filepath
    stat_dict['model'] = model_path
    return stat_dict

def get_stats_from_model_args(kwargs):
    return get_stats_from_model(**kwargs)

def gather_stats(network: str, folder: str, dataset: str):
    models = []
    
    with Pool() as p:
        files = os.listdir(folder)
        model_checkpoints = [os.path.join(folder, filename) for filename in files]
        model_checkpoints = list(filter(lambda x: x.endswith(".txt"), model_checkpoints))
        model_checkpoints = [{
            'network': network,
            'filepath': checkpoint,
            'dataset': dataset,
            'batch': 128,
            'prune_mode': 'random',
            'neurons': 500,
            'seed': 42
        } for checkpoint in model_checkpoints]
        models = p.map(get_stats_from_model_args, model_checkpoints)

    return models

# %% Save minimization vs maximization
def save_min_max(network_type: str, folder_max: str, folder_min: str, dataset: str, output: str):
    loader = get_loader(network_type)
    mx = gather_stats(loader, folder_max, dataset)
    mn = gather_stats(loader, folder_min, dataset)
    
    result = {
        "mx": mx,
        "mn": mn
    }

    with open(output, "w") as f:
        f.write(json.dumps(result, indent=4))
    print(f"JSON output saved at {output}")

# %% Process Statistics
def prune(network_type: str, checkpoint: str, dataset: str, prune_mode: str, output: str, json_output: str, tries: int = 1):
    print(f"Test pruning on {network_type}...")
    NEURONS = [2000, 1500, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
    # NEURONS = [2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400, 200]
    loader = get_loader(network_type)

    torch.manual_seed(123)
    random_seeds = torch.randint(0, int(1e9), (tries, ))
    
    result = { str(NEURONS[i]) : [] for i in range(len(NEURONS)) }
    for t in range(tries):
        args = [{
            'network': loader,
            'filepath': checkpoint,
            'dataset': dataset,
            'batch': 256,
            'prune_mode': prune_mode,
            'neurons': neurons,
            'output': output,
            'seed': random_seeds[t].item()
        } for neurons in NEURONS]
        
        with Pool(4) as p:
            data = p.map(get_stats_from_model_args, args)
            for i in range(len(data)):
                result[str(NEURONS[i])].append(data[i])
    
    with open(json_output, "w") as f:
        f.write(json.dumps(result, indent=4))
    print(f"JSON output saved at {json_output}")

# %% Accuracy report
def print_acc_to_json(main_dir: str, dataset: str, network_type: str, output_dir: str):
    loader = get_loader(network_type)

    os.makedirs(output_dir, exist_ok=True)
    dir = f"{main_dir}/models/{dataset}/{network_type}"
    checkpoints = []
    for file in os.listdir(dir):
        if file.endswith(".txt"):
            checkpoint = os.path.join(dir, file)
            checkpoints.append(checkpoint)

    checkpoint_args = []
    for checkpoint in checkpoints:
        print(f"Test accuracy on {network_type} {checkpoint}...")
        args = {
            'network': loader,
            'filepath': checkpoint,
            'dataset': dataset,
            'batch': 256,
            'prune_mode': 'random',
            'neurons': 2000,
            'seed': 42,
            'output': os.path.join(output_dir, f"{network_type}_{dataset}.pt")
        }

        checkpoint_args.append(args)
        
    with Pool(4) as p:
        data = p.map(get_stats_from_model_args, checkpoint_args)

    output_json = os.path.join(output_dir, f"{network_type}_{dataset}.json")
    result = { 'network': network_type, 'dataset': dataset, 'data': data }
    with open(output_json, "w") as f:
        f.write(json.dumps(result, indent=4))
    print(f"JSON output saved at {output_json}")


# %% Make LaTeX table for accuracy report
def make_acc_latex_table():
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Baseline test accuracy (\\%) across different networks and datasets.}")
    print("             & & \\multicolumn{3}{c}{\\textbf{BPNN}} & \\multicolumn{3}{c}{\\textbf{FFNN}} & \\multicolumn{3}{c}{\\textbf{FFNN+C}} & \\multicolumn{3}{c}{\\textbf{FFRNN}} \\")
    print("\\begin{tabularx}{\\textwidth}{LCCC}")
    print("\\toprule")
    print("\\textbf{Network} & \\textbf{MNIST} & \\textbf{FashionMNIST} & \\textbf{CIFAR-10} \\\\")
    print("\\midrule")

    for net in ['bp', 'ff', 'ffc', 'ffrnn']:
        print(f"{DISPLAY_NAMES[net]}", end="")
        for dataset in ['mnist', 'fashion', 'cifar10']:
            with open(f"./accuracy_reports/{net}_{dataset}.json", "r") as f:
                data = json.load(f)
                accs = [round(d['pretest'] * 100, 2) for d in data['data']]
                std = np.std(accs)
                accs_str = f"{np.mean(accs):.2f} $\\pm$ {std:.2f}"
                print(f" & {accs_str}", end="")
        print("\\\\")

    print("\\bottomrule")
    print("\\end{tabularx}")
    print("\\end{table}")

# %% Make LaTeX table for sparsity report
def make_sparsity_latex_table():
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Sensitivity analysis of the baseline models' accuracy across different learning rates, and loss thresholds.}")
    print("\\label{table:sparsity_baseline}")
    print("\\begin{tabularx}{\\textwidth}{lcccCC}")
    print("\\toprule")
    print("\\textbf{ST} & \\textbf{Network} & \\textbf{Layer} & \\textbf{MNIST} & \\textbf{FashionMNIST} & \\textbf{CIFAR-10} \\\\")
    print("\\midrule")

    for sparsity in [SparsityType.HOYER, SparsityType.L1_NEG_ENTROPY, SparsityType.L2_NEG_ENTROPY, SparsityType.GINI]:
        print(f"\\multirow{{5}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\textbf{str(sparsity).split('.')[1]}}}}}", end="")
        for net in ['bp', 'ff']:
            if net == 'ff':
                print(f"& \\multirow{{2}}{{*}}{{{DISPLAY_NAMES[net]}}}", end="")
                for layer in [0, 1]:
                    print(f" & \\textbf{{Layer {layer + 1}}}", end="")
                    for dataset in ['mnist', 'fashion', 'cifar10']:
                        bp, ff, ffrnn = accumulate_sparsity_report(dataset, sparsity)
                        mean = np.mean(ff[layer], axis=0)
                        std = np.std(ff[layer], axis=0)
                        sparsity_str = f"{mean[-1]:.7f} $\\pm$ {std[-1]:.7f}"
                        print(f" & {sparsity_str}", end="")
                    print("\\\\")
            elif net == 'bp':
                print(f"& \\multirow{{3}}{{*}}{{{DISPLAY_NAMES[net]}}}", end="")
                for layer in [0, 1, 2]:
                    print(f" & \\textbf{{Layer {layer + 1}}}", end="")
                    for dataset in ['mnist', 'fashion', 'cifar10']:
                        bp, ff, ffrnn = accumulate_sparsity_report(dataset, sparsity)
                        mean = np.mean(bp[layer], axis=0)
                        std = np.std(bp[layer], axis=0)
                        sparsity_str = f"{mean[-1]:.7f} $\\pm$ {std[-1]:.7f}"
                        print(f" & {sparsity_str}", end="")
                    print("\\\\")
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabularx}")
    print("\\end{table}")

def make_sensitivity_analysis_latex_table(main_dir: str):
    # Columns: Learning Rate | Loss Threshold | (MNIST | FashionMNIST | CIFAR-10)
    # Accuracy for each dataset across different networks, learning rates, and loss thresholds
    print("\\begin{table}[H]")
    print("    \\centering")
    print("    \\caption{Sensitivity analysis of the baseline models across different networks, learning rates, and loss thresholds.}")
    print("    \\begin{adjustwidth}{-\\extralength}{0cm}")
    print("        \\label{table:sensitivity_analysis}")
    print("        \\begin{tabularx}{\\fulllength}{lCccccccccccccc}")
    print("            \\toprule")
    print("             & & \\multicolumn{3}{c}{\\textbf{BPNN}} & \\multicolumn{3}{c}{\\textbf{FFNN}} & \\multicolumn{3}{c}{\\textbf{FFNN+C}} & \\multicolumn{3}{c}{\\textbf{FFRNN}} \\\\")
    print("            \\textbf{LR} & \\textbf{LT} & \\textbf{M} & \\textbf{F} & \\textbf{C} & \\textbf{M} & \\textbf{F} & \\textbf{C} & \\textbf{M} & \\textbf{F} & \\textbf{C} & \\textbf{M} & \\textbf{F} & \\textbf{C} \\\\")
    print("            \\midrule")

    accuracy_reports = { }
    for net in ['bp', 'ff', 'ffc', 'ffrnn']:
        for dataset in ['mnist', 'fashion', 'cifar10']:
            with open(f"{main_dir}/{net}_{dataset}.json", "r") as f:
                data = json.load(f)
                for d_obj in data['data']:
                    filename = d_obj['report']
                    filename = os.path.basename(filename)
                    parts = filename.split('_')
                    lr = float(parts[0])
                    lt = float(parts[1])
                    mean = d_obj['pretest'] * 100
                    if (lr, lt) not in accuracy_reports:
                        accuracy_reports[(lr, lt)] = {}
                    if net not in accuracy_reports[(lr, lt)]:
                        accuracy_reports[(lr, lt)][net] = {}
                    accuracy_reports[(lr, lt)][net][dataset] = mean

    # Sort by lr, lt
    for lr in sorted(set(k[0] for k in accuracy_reports.keys())):
        for lt in sorted(set(k[1] for k in accuracy_reports.keys())):
            key = (lr, lt)
            if key not in accuracy_reports:
                continue
            print(f"            {lr} & {lt}", end="")
            for net in ['bp', 'ff', 'ffc', 'ffrnn']:
                for dataset in ['mnist', 'fashion', 'cifar10']:
                    val = accuracy_reports[key].get(net, {}).get(dataset, None)
                    if val is not None:
                        print(f" & {val:.2f}", end="")
                    else:
                        print(" & -", end="")
            print(" \\\\")
    print("            \\bottomrule")
    print("        \\end{tabularx}")
    print("    \\end{adjustwidth}")
    print("\\end{table}")

def make_sensitivity_analysis_sparsity_latex_table(main_dir: str):
    # Columns: Learning Rate | Loss Threshold | (MNIST | FashionMNIST | CIFAR-10)
    # Sparsity for each dataset across different networks, learning rates, and loss thresholds
    print("\\begin{table}[H]")
    print("    \\centering")
    print("    \\caption{Sparsity analysis of the baseline models across different networks, learning rates, and loss thresholds.}")
    print("    \\begin{adjustwidth}{-\\extralength}{0cm}")
    print("        \\label{table:sensitivity_analysis}")
    print("        \\begin{tabularx}{\\fulllength}{lCccccccccccccc}")
    print("            \\toprule")
    print("             & & \\multicolumn{3}{c}{\\textbf{BPNN}} & \\multicolumn{3}{c}{\\textbf{FFNN}} & \\multicolumn{3}{c}{\\textbf{FFNN+C}} & \\multicolumn{3}{c}{\\textbf{FFRNN}} \\\\")
    print("            \\textbf{LR} & \\textbf{LT} & \\textbf{M} & \\textbf{F} & \\textbf{C} & \\textbf{M} & \\textbf{F} & \\textbf{C} & \\textbf{M} & \\textbf{F} & \\textbf{C} & \\textbf{M} & \\textbf{F} & \\textbf{C} \\\\")
    print("            \\midrule")

    sparsity_report = { }
    for net in ['bp', 'ff', 'ffc', 'ffrnn']:
        for dataset in ['mnist', 'fashion', 'cifar10']:
            with open(f"{main_dir}/{net}_{dataset}.json", "r") as f:
                data = json.load(f)
                for d_obj in data['data']:
                    hoyer = d_obj['HOYER']

                    layer = []
                    if net == 'ff':
                        layer.append(hoyer['layer_0'])
                        layer.append(hoyer['layer_1'])
                    if net == 'ffc':
                        layer.append(hoyer['layer_0'])
                        layer.append(hoyer['layer_1'])
                    if net == 'ffrnn':
                        layer.append(hoyer['layer_1']['fw+bw'])
                        layer.append(hoyer['layer_2']['fw+bw'])
                    if net == 'bp':
                        layer.append(hoyer['layer_0']['weight'])
                        layer.append(hoyer['layer_1']['weight'])
                        layer.append(hoyer['layer_2']['weight'])
                    
                    mean = np.nanmean(layer)

                    # Get LR and LT from the filename
                    filename = d_obj['report']
                    filename = os.path.basename(filename)
                    parts = filename.split('_')
                    lr = float(parts[0])
                    lt = float(parts[1])
                    if (lr, lt) not in sparsity_report:
                        sparsity_report[(lr, lt)] = {}
                    if net not in sparsity_report[(lr, lt)]:
                        sparsity_report[(lr, lt)][net] = {}
                    sparsity_report[(lr, lt)][net][dataset] = mean

    # Sort by lr, lt
    for lr in sorted(set(k[0] for k in sparsity_report.keys())):
        for lt in sorted(set(k[1] for k in sparsity_report.keys())):
            key = (lr, lt)
            if key not in sparsity_report:
                continue
            print(f"            {lr} & {lt}", end="")
            for net in ['bp', 'ff', 'ffc', 'ffrnn']:
                for dataset in ['mnist', 'fashion', 'cifar10']:
                    val = sparsity_report[key].get(net, {}).get(dataset, None)
                    if val is not None:
                        print(f" & {val:.2f}", end="")
                    else:
                        print(" & -", end="")
            print(" \\\\")
    print("            \\bottomrule")
    print("        \\end{tabularx}")
    print("    \\end{adjustwidth}")
    print("\\end{table}")

def make_sensitivity_analysis_plot(main_dir: str, output_dir: str):
    # For each of the datasets and for each network, plots a confusion-like matrix
    # where the x-axis is the learning rate, y-axis is the loss threshold, and the color is the accuracy
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    datasets = ['mnist', 'fashion', 'cifar10']
    networks = ['bp', 'ff', 'ffc', 'ffrnn']

    plt.rcParams.update({'font.size': 14})

    for dataset in datasets:
        fig, axes = plt.subplots(1, 5, figsize=(30, 6), sharey=True)
        plt.subplots_adjust(wspace=0.2)
        vmin, vmax = 0, 100
        unique_lrs, unique_lts = None, None
        ims = []
        for idx, network in enumerate(networks):
            with open(f"{main_dir}/{network}_{dataset}.json", "r") as f:
                data = json.load(f)
                accuracies = []
                lrs = []
                lts = []
                for d_obj in data['data']:
                    filename = d_obj['report']
                    filename = os.path.basename(filename)
                    lr = float(filename.split('_')[0])
                    lt = float(filename.split('_')[1])
                    acc = d_obj['pretest'] * 100
                    lrs.append(lr)
                    lts.append(lt)
                    accuracies.append(acc)
                unique_lrs = sorted(set(lrs))
                unique_lts = sorted(set(lts))
                grid = np.zeros((len(unique_lts), len(unique_lrs)))
                for i, (lr, lt, acc) in enumerate(zip(lrs, lts, accuracies)):
                    x_idx = unique_lrs.index(lr)
                    y_idx = unique_lts.index(lt)
                    grid[y_idx, x_idx] = acc
                ax = axes[idx]
                im = ax.imshow(grid, aspect='equal', cmap='viridis', origin='lower',
                               norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
                ims.append(im)
                # Draw numbers on the heatmap
                for i in range(len(unique_lts)):
                    for j in range(len(unique_lrs)):
                        acc_val = grid[i, j]
                        if acc_val > 0:
                            color = 'black' if acc_val >= 50 else 'white'
                            ax.text(j, i, f"{acc_val:.1f}", ha='center', va='center', color=color, fontsize=14)
                ax.set_xticks(np.arange(len(unique_lrs)))
                ax.set_xticklabels([f"{lr:g}" for lr in unique_lrs])
                ax.set_yticks(np.arange(len(unique_lts)))
                ax.set_yticklabels([f"{lt:g}" for lt in unique_lts])
                ax.set_ylabel('Loss Threshold')
                ax.set_xlabel('Learning Rate')
                ax.set_title(DISPLAY_NAMES.get(network, network))
                ax.grid(False)
        # Add a single colorbar to the last (5th) axis
        cbar_ax = axes[-1]
        fig.delaxes(cbar_ax)  # Remove the 5th axis to use as colorbar
        cbar = fig.colorbar(ims[-1], ax=axes[:-1], orientation='vertical', fraction=0.025, pad=0.04)
        cbar.set_label('Accuracy (%)')
        output_file = f"{output_dir}/{dataset}_accuracy_heatmap.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Heatmap saved at {output_file}")
    
def make_sensitivity_analysis_sparsity_plot(main_dir: str, output_dir: str):
    # For each of the datasets and for each network, plots a confusion-like matrix
    # where the x-axis is the learning rate, y-axis is the loss threshold, and the color is the sparsity
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    datasets = ['mnist', 'fashion', 'cifar10']
    networks = ['bp', 'ff', 'ffc', 'ffrnn']

    plt.rcParams.update({'font.size': 14})

    def get_sparsity_from_json(network, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
            sparsities = []
            lrs = []
            lts = []
            for d_obj in data['data']:
                filename = d_obj['report']
                filename = os.path.basename(filename)
                lr = float(filename.split('_')[0])
                lt = float(filename.split('_')[1])
                
                hoyer = d_obj['HOYER']
                layer = []
                if network == 'ff':
                    layer.append(hoyer['layer_0'])
                    layer.append(hoyer['layer_1'])
                if network == 'ffc':
                    layer.append(hoyer['layer_0'])
                    layer.append(hoyer['layer_1'])
                if network == 'ffrnn':
                    layer.append(hoyer['layer_1']['fw+bw'])
                    layer.append(hoyer['layer_2']['fw+bw'])
                if network == 'bp':
                    layer.append(hoyer['layer_0']['weight'])
                    layer.append(hoyer['layer_1']['weight'])
                    layer.append(hoyer['layer_2']['weight'])
                    
                sparsity = np.nanmean(layer) if len(layer) > 0 else 0.5

                lrs.append(lr)
                lts.append(lt)
                sparsities.append(sparsity)
            return sparsities, lrs, lts

    for dataset in datasets:
        fig, axes = plt.subplots(1, 5, figsize=(30, 6), sharey=True)
        plt.subplots_adjust(wspace=0.2)
        vmin, vmax = 0.0, 1.0
        unique_lrs, unique_lts = None, None
        ims = []
        for idx, network in enumerate(networks):
            sparsities, lrs, lts = get_sparsity_from_json(network, f"{main_dir}/{network}_{dataset}.json")
            unique_lrs = sorted(set(lrs))
            unique_lts = sorted(set(lts))
            grid = np.zeros((len(unique_lts), len(unique_lrs)))
            for i, (lr, lt, sparsity) in enumerate(zip(lrs, lts, sparsities)):
                x_idx = unique_lrs.index(lr)
                y_idx = unique_lts.index(lt)
                grid[y_idx, x_idx] = sparsity
            ax = axes[idx]
            im = ax.imshow(grid, aspect='equal', cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            ims.append(im)
            # Draw numbers on the heatmap
            for i in range(len(unique_lts)):
                for j in range(len(unique_lrs)):
                    sparsity_value = grid[i, j]
                    if sparsity_value > 0:
                        color = 'black' if sparsity_value >= 0.5 else 'white'
                        ax.text(j, i, f"{sparsity_value:.2f}", ha='center', va='center', color=color, fontsize=14)
            ax.set_xticks(np.arange(len(unique_lrs)))
            ax.set_xticklabels([f"{lr:g}" for lr in unique_lrs])
            ax.set_yticks(np.arange(len(unique_lts)))
            ax.set_yticklabels([f"{lt:g}" for lt in unique_lts])
            ax.set_ylabel('Loss Threshold')
            ax.set_xlabel('Learning Rate')
            ax.set_title(DISPLAY_NAMES.get(network, network))
            ax.grid(False)
        # Add a single colorbar to the last (5th) axis
        cbar_ax = axes[-1]
        fig.delaxes(cbar_ax)  # Remove the 5th axis to use as colorbar
        cbar = fig.colorbar(ims[-1], ax=axes[:-1], orientation='vertical', fraction=0.025, pad=0.04)
        cbar.set_label('Hoyer Sparsity')
        output_file = f"{output_dir}/{dataset}_sparsity_heatmap.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Sparsity Heatmap saved at {output_file}")

# %% Run
if __name__ == '__main__':
    # output_dir = './accuracy_reports_4'
    # for net in ['bp', 'ff', 'ffc', 'ffrnn']:
    #     for dataset in ['mnist', 'fashion', 'cifar10']:
    #         print_acc_to_json("./acc_3", dataset, net, output_dir)
    #         for file in os.listdir(output_dir):
    #             if file.endswith('.pt'):
    #                 os.remove(os.path.join(output_dir, file))

    # make_sensitivity_analysis_plot('./accuracy_reports_4', './sensitivity_analysis_accuracy_plots')
    # make_sensitivity_analysis_sparsity_plot('./accuracy_reports_4', './sensitivity_analysis_sparsity_plots')

    # make_acc_latex_table()
    # make_sparsity_latex_table()

    # main_dir = "./accuracy_reports_3"
    # make_sensitivity_analysis_latex_table(main_dir)
    # make_sensitivity_analysis_sparsity_latex_table(main_dir)

    # The difference between minimization and maximization
    # save_min_max('ff3', './models_ff_v3_max', './models_ff_v3_min', 'mnist', './minimize_reports2/ff_v3.json')
    # save_min_max('ff', './models_ff_max', './models_ff_min', 'mnist', './minimize_reports2/ff.json')
    # save_min_max('ffc', './models_ff_c_max', './models_ff_c_min', 'mnist', './minimize_reports2/ffc.json')
    # save_min_max('ffrnn', './models_ff_rnn_max', './models_ff_rnn_min', 'mnist', './minimize_reports2/ffrnn.json')

    # MNIST
    # dataset = 'mnist'
    # prune_mode = 'random'
    # prune_reports_out = 'prune_reports_min_10t'
    # pruned_models_out_dir = 'prune_models_min_10t'
    # models = {
    #     'bp': './models_bp/bp_mnist_3773ec.pt',
    #     'ff': './models_ff_min/ff_min_f93710.pt',
    #     'ffc': './models_ff_c_min/ff_c_min_a59e32.pt',
    #     'ffrnn': './models_ff_rnn_min/ff_rnn_min_98c25b.pt',
    # }

    # prune('bp', models['bp'], dataset, prune_mode, f'./{pruned_models_out_dir}/bp.pt', f'./{prune_reports_out}/bp.json', 10)
    # prune('ff', models['ff'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_min.pt', f'./{prune_reports_out}/ff.json', 10)
    # prune('ffc', models['ffc'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_c_min.pt', f'./{prune_reports_out}/ffc.json', 10)
    # prune('ffrnn', models['ffrnn'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_rnn_min.pt', f'./{prune_reports_out}/ffrnn.json', 10)

    # dataset = 'mnist'
    # prune_mode = 'last'
    # prune_reports_out = 'prune_reports_max_last_10t'
    # pruned_models_out_dir = 'prune_models_max_last_10t'
    # tries = 10
    # models = {
    #     'bp': './models_bp/bp_mnist_3773ec.pt',
    #     'ff': './models_ff_max/ff_max_c9e323.pt',
    #     'ffc': './models_ff_c_max/ff_c_max_764146.pt',
    #     'ffrnn': './models_ff_rnn_max/ff_rnn_max_aa035a.pt',
    # }

    # prune('bp', models['bp'], dataset, prune_mode, f'./{pruned_models_out_dir}/bp.pt', f'./{prune_reports_out}/bp.json', tries)
    # prune('ff', models['ff'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_min.pt', f'./{prune_reports_out}/ff.json', tries)
    # prune('ffc', models['ffc'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_c_min.pt', f'./{prune_reports_out}/ffc.json', tries)
    # prune('ffrnn', models['ffrnn'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_rnn_min.pt', f'./{prune_reports_out}/ffrnn.json', tries)

    # # MNSIT MIN FIRST
    # dataset = 'mnist'
    # prune_mode = 'first'
    # prune_reports_out = 'prune_reports_min_first_10t'
    # pruned_models_out_dir = 'prune_models_min_first_10t'
    # tries = 1
    # models = {
    #     'bp': './models_bp/bp_mnist_3773ec.pt',
    #     'ff': './models_ff_min/ff_min_f93710.pt',
    #     'ffc': './models_ff_c_min/ff_c_min_a59e32.pt',
    #     'ffrnn': './models_ff_rnn_min/ff_rnn_min_98c25b.pt',
    # }

    # os.makedirs(prune_reports_out, exist_ok=True)
    # os.makedirs(pruned_models_out_dir, exist_ok=True)

    # prune('bp', models['bp'], dataset, prune_mode, f'./{pruned_models_out_dir}/bp.pt', f'./{prune_reports_out}/bp.json', tries)
    # prune('ff', models['ff'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_min.pt', f'./{prune_reports_out}/ff.json', tries)
    # prune('ffc', models['ffc'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_c_min.pt', f'./{prune_reports_out}/ffc.json', tries)
    # prune('ffrnn', models['ffrnn'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_rnn_min.pt', f'./{prune_reports_out}/ffrnn.json', tries)

    # # MNSIT TW
    # dataset = 'mnist'
    # prune_mode = 'threshold-weights'
    # prune_reports_out = 'prune_reports_min_tw3_1t'
    # pruned_models_out_dir = 'prune_models_min_tw3_1t'
    # # models = {
    # #     'bp': './models_bp/bp_mnist_3773ec.pt',
    # #     'ff': './models_ff_max/ff_max_c9e323.pt',
    # #     'ffc': './models_ff_c_max/ff_c_max_764146.pt',
    # #     'ffrnn': './models_ff_rnn_max/ff_rnn_max_aa035a.pt',
    # # }
    # tries = 1
    # models = {
    #     'bp': './models_bp/bp_mnist_3773ec.pt',
    #     'ff': './models_ff_min/ff_min_f93710.pt',
    #     'ffc': './models_ff_c_min/ff_c_min_a59e32.pt',
    #     'ffrnn': './models_ff_rnn_min/ff_rnn_min_98c25b.pt',
    # }

    # os.makedirs(prune_reports_out, exist_ok=True)
    # os.makedirs(pruned_models_out_dir, exist_ok=True)

    # prune('bp', models['bp'], dataset, prune_mode, f'./{pruned_models_out_dir}/bp.pt', f'./{prune_reports_out}/bp.json', tries)
    # prune('ff', models['ff'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_min.pt', f'./{prune_reports_out}/ff.json', tries)
    # prune('ffc', models['ffc'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_c_min.pt', f'./{prune_reports_out}/ffc.json', tries)
    # prune('ffrnn', models['ffrnn'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_rnn_min.pt', f'./{prune_reports_out}/ffrnn.json', tries)

    # # MNSIT TW
    # dataset = 'mnist'
    # prune_mode = 'random-weights'
    # prune_reports_out = 'prune_reports_max_rw_1t'
    # pruned_models_out_dir = 'prune_models_max_rw_1t'
    # models = {
    #     'bp': './models_bp/bp_mnist_3773ec.pt',
    #     'ff': './models_ff_max/ff_max_c9e323.pt',
    #     'ffc': './models_ff_c_max/ff_c_max_764146.pt',
    #     'ffrnn': './models_ff_rnn_max/ff_rnn_max_aa035a.pt',
    # }
    # tries = 1
    # # models = {
    # #     'bp': './models_bp/bp_mnist_3773ec.pt',
    # #     'ff': './models_ff_min/ff_min_f93710.pt',
    # #     'ffc': './models_ff_c_min/ff_c_min_a59e32.pt',
    # #     'ffrnn': './models_ff_rnn_min/ff_rnn_min_98c25b.pt',
    # # }

    # os.makedirs(prune_reports_out, exist_ok=True)
    # os.makedirs(pruned_models_out_dir, exist_ok=True)

    # prune('bp', models['bp'], dataset, prune_mode, f'./{pruned_models_out_dir}/bp.pt', f'./{prune_reports_out}/bp.json', tries)
    # prune('ff', models['ff'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_min.pt', f'./{prune_reports_out}/ff.json', tries)
    # prune('ffc', models['ffc'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_c_min.pt', f'./{prune_reports_out}/ffc.json', tries)
    # prune('ffrnn', models['ffrnn'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_rnn_min.pt', f'./{prune_reports_out}/ffrnn.json', tries)


    # Prune one of the models
    # prune_mode = 'last'
    # out_prefix = 'max_last'
    # prune('ff', './models_ff_max/ff_max_c9e323.pt', 'mnist', prune_mode, './models_ff_max_out/ff_max.pt', f'./prune_reports_{out_prefix}/ff.json')
    # prune('ffc', './models_ff_c_max/ff_c_max_764146.pt', 'mnist', prune_mode, './models_ff_c_max_out/ff_c_max.pt', f'./prune_reports_{out_prefix}/ffc.json')
    # prune('ffrnn', './models_ff_rnn_max/ff_rnn_max_aa035a.pt', 'mnist', prune_mode, './models_ff_rnn_max_out/ff_rnn_max.pt', f'./prune_reports_{out_prefix}/ffrnn.json')
    # prune('bp', './models_bp/bp_mnist_3773ec.pt', 'mnist', prune_mode, './models_bp_out/bp.pt', f'./prune_reports_{out_prefix}/bp.json')

    # prune('ff', './models_ff_min/ff_min_f93710.pt', 'mnist', prune_mode, './models_ff_min_out/ff_min.pt', f'./prune_reports_{out_prefix}/ff.json')
    # prune('ffc', './models_ff_c_min/ff_c_min_a59e32.pt', 'mnist', prune_mode, './models_ff_c_min_out/ff_c_min.pt', f'./prune_reports_{out_prefix}/ffc.json')
    # prune('ffrnn', './models_ff_rnn_min/ff_rnn_min_98c25b.pt', 'mnist', prune_mode, './models_ff_rnn_min_out/ff_rnn_min.pt', f'./prune_reports_{out_prefix}/ffrnn.json')
    # prune('bp', './models_bp/bp_mnist_3773ec.pt', 'mnist', prune_mode, './models_bp_out/bp.pt', f'./prune_reports_{out_prefix}/bp.json')
    
    # CIFAR10
    # dataset = 'cifar10'
    # prune_mode = 'random'
    # prune_reports_out = 'prune_reports_cifar10_2_10t'
    # pruned_models_out_dir = 'prune_models_cifar10_2_10t'
    # models = {
    #     'bp': 'bp_ef804e.pt',
    #     'ff': 'ff_79499d.pt',
    #     'ffc': 'ffc_b19697.pt',
    #     'ffrnn': 'ffrnn_8310db.pt',
    # }

    # os.makedirs(prune_reports_out, exist_ok=True)
    # os.makedirs(pruned_models_out_dir, exist_ok=True)

    # for key in models.keys():
    #     models[key] = './models_cifar_ff_2/' + models[key]

    # # prune('bp', models['bp'], dataset, prune_mode, f'./{pruned_models_out_dir}/bp.pt', f'./{prune_reports_out}/bp.json', 10)
    # # prune('ff', models['ff'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_min.pt', f'./{prune_reports_out}/ff.json', 10)
    # # prune('ffc', models['ffc'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_c_min.pt', f'./{prune_reports_out}/ffc.json', 10)
    # prune('ffrnn', models['ffrnn'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_rnn_min.pt', f'./{prune_reports_out}/ffrnn.json', 10)

    # # FASHION MNIST
    # dataset = 'fashion'
    # prune_mode = 'random'
    # prune_reports_out = 'prune_reports_fashion_10t'
    # pruned_models_out_dir = 'prune_models_fashion_10t'
    # tries = 10
    # models = {
    #     'bp': 'bp_2d07dc.pt',
    #     'ff': 'ff_be52ce.pt',
    #     'ffc': 'ffc_b11968.pt',
    #     'ffrnn': 'ffrnn_8e7bd9.pt',
    # }

    # for key in models.keys():
    #     models[key] = './models_fashion_ff/' + models[key]

    # prune('bp', models['bp'], dataset, prune_mode, f'./{pruned_models_out_dir}/bp.pt', f'./{prune_reports_out}/bp.json', tries)
    # prune('ff', models['ff'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_min.pt', f'./{prune_reports_out}/ff.json', tries)
    # prune('ffc', models['ffc'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_c_min.pt', f'./{prune_reports_out}/ffc.json', tries)
    # prune('ffrnn', models['ffrnn'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_rnn_min.pt', f'./{prune_reports_out}/ffrnn.json', tries)

    # FASHION MNIST
    # dataset = 'fashion'
    # prune_mode = 'random'
    # prune_reports_out = 'prune_reports_fashion'
    # pruned_models_out_dir = 'prune_models_fashion'
    # models = {
    #     'bp': 'bp_2d07dc.pt',
    #     'ff': 'ff_be52ce.pt',
    #     'ffc': 'ffc_b11968.pt',
    #     'ffrnn': 'ffrnn_8e7bd9.pt',
    # }

    # for key in models.keys():
    #     models[key] = './models_fashion_ff/' + models[key]

    # prune('bp', models['bp'], dataset, prune_mode, f'./{pruned_models_out_dir}/bp.pt', f'./{prune_reports_out}/bp.json')
    # prune('ff', models['ff'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_min.pt', f'./{prune_reports_out}/ff.json')
    # prune('ffc', models['ffc'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_c_min.pt', f'./{prune_reports_out}/ffc.json')
    # prune('ffrnn', models['ffrnn'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_rnn_min.pt', f'./{prune_reports_out}/ffrnn.json')

    # # MNSIT TW
    # dataset = 'mnist'
    # prune_mode = 'invert-ends'
    # prune_reports_out = 'prune_reports_min_ie_1t'
    # pruned_models_out_dir = 'prune_models_min_ie_1t'
    # # models = {
    # #     'bp': './models_bp/bp_mnist_3773ec.pt',
    # #     'ff': './models_ff_max/ff_max_c9e323.pt',
    # #     'ffc': './models_ff_c_max/ff_c_max_764146.pt',
    # #     'ffrnn': './models_ff_rnn_max/ff_rnn_max_aa035a.pt',
    # # }
    # tries = 1
    # models = {
    #     'bp': './models_bp/bp_mnist_3773ec.pt',
    #     'ff': './models_ff_min/ff_min_f93710.pt',
    #     'ffc': './models_ff_c_min/ff_c_min_a59e32.pt',
    #     'ffrnn': './models_ff_rnn_min/ff_rnn_min_98c25b.pt',
    # }

    # os.makedirs(prune_reports_out, exist_ok=True)
    # os.makedirs(pruned_models_out_dir, exist_ok=True)

    # prune('bp', models['bp'], dataset, prune_mode, f'./{pruned_models_out_dir}/bp.pt', f'./{prune_reports_out}/bp.json', tries)
    # prune('ff', models['ff'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_min.pt', f'./{prune_reports_out}/ff.json', tries)
    # prune('ffc', models['ffc'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_c_min.pt', f'./{prune_reports_out}/ffc.json', tries)
    # prune('ffrnn', models['ffrnn'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_rnn_min.pt', f'./{prune_reports_out}/ffrnn.json', tries)



## RNN
# tensor([0.9788, 0.9792, 0.9784, 0.9756, 0.9794, 0.9772, 0.9774, 0.9790, 0.9773])
# tensor([0.9837, 0.9824, 0.9823, 0.9834, 0.9835, 0.9829, 0.9825, 0.9817, 0.9844,
#         0.9812])
# tensor(0.9780) tensor(0.9828)
# tensor(0.0012) tensor(0.0010)
# tensor(0.9794) tensor(0.9844)
