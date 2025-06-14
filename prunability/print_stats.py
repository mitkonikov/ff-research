# %% Imports
import torch
import os
import asyncio
import json
import torch

from multiprocessing import Pool

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
        '-n', str(neurons),
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

# %% Run
if __name__ == '__main__':
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
    dataset = 'cifar10'
    prune_mode = 'random'
    prune_reports_out = 'prune_reports_cifar10_2_10t'
    pruned_models_out_dir = 'prune_models_cifar10_2_10t'
    models = {
        'bp': 'bp_ef804e.pt',
        'ff': 'ff_79499d.pt',
        'ffc': 'ffc_b19697.pt',
        'ffrnn': 'ffrnn_8310db.pt',
    }

    os.makedirs(prune_reports_out, exist_ok=True)
    os.makedirs(pruned_models_out_dir, exist_ok=True)

    for key in models.keys():
        models[key] = './models_cifar_ff_2/' + models[key]

    # prune('bp', models['bp'], dataset, prune_mode, f'./{pruned_models_out_dir}/bp.pt', f'./{prune_reports_out}/bp.json', 10)
    # prune('ff', models['ff'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_min.pt', f'./{prune_reports_out}/ff.json', 10)
    # prune('ffc', models['ffc'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_c_min.pt', f'./{prune_reports_out}/ffc.json', 10)
    prune('ffrnn', models['ffrnn'], dataset, prune_mode, f'./{pruned_models_out_dir}/ff_rnn_min.pt', f'./{prune_reports_out}/ffrnn.json', 10)

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
# %%
