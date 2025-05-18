# %% Imports
from utils import ArgumentsLoad
# args = ArgumentsLoad()(['-i', './models_ff_rnn_max/ff_rnn_max_aa035a.pt'])
# args = ArgumentsLoad()(['-i', './models_ff_rnn_min/ff_rnn_min_98c25b.pt'])
args = ArgumentsLoad()()

import torch
import matplotlib.pyplot as plt

from fflib.utils.data.mnist import NegativeGenerator as MNISTNEG
from fflib.utils.data.datasets import CreateDatasetFromName
from fflib.nn.ff_rnn import FFRNN
from fflib.probes.one_hot import TryAllClasses
from fflib.utils.ffrnn_suite import FFRNNSuite
from fflib.enums import SparsityType
from fflib.utils.ff_logger import logger

from utils import file_size, prune, plot_activations
from typing import cast

logger.disabled = not args.verbose

# Setup the device
device_type = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device type: {device_type}")
device = torch.device(device_type)

torch.manual_seed(args.seed)

# %% Statistics Output Dictionary
stats = { }

# %% Setup Dataset
logger.info("Setting up the dataset...")
dataloader = CreateDatasetFromName(
    name=args.dataset,
    batch_size=args.batch,
    validation_split=0.1,
    negative_generator=MNISTNEG.RANDOM,
)

# %% Probe
logger.info("Setting up a probe...")
probe = TryAllClasses(lambda x, y: suite.net(x, y), output_classes=10)

# %% Create Test Suite
logger.info("Setting up a TestSuite...")
suite = FFRNNSuite(args.input, probe, dataloader, device)
probe.maximize = cast(FFRNN, suite.net).layers[1].maximize
stats["original_size"] = file_size(args.input)

# %% Run Test
if args.pretest:
    stats["pretest"] = suite.test()

# %%
net = cast(FFRNN, suite.net)

activations = []
for i in range(8):
    activations.append([torch.Tensor() for _ in range(len(net.layers))])

def save_activation(x: torch.Tensor, id: int, frame: int):
    activations[frame][id] = x.clone().detach()

net.register_hook('layer_activation', 'h1', save_activation)

# activations => [frame, layer, batch_size, neurons]

# Take one batch
it = iter(dataloader.test_loader)
batch = next(it)
x, y = batch[0].to(device), batch[1].to(device)

y = dataloader.encode_output(y)

for i in range(8):
    activations[i][0] = x.detach()
    activations[i][-1] = y.detach()

net(x, y)

net._create_hooks_dict()

# %% Plot weights
if args.plot:
    unflatten = torch.nn.Unflatten(0, (28, 28))
    unflatten_layer = torch.nn.Unflatten(0, (50, 40))

    fig, axes = plt.subplots(8, len(net.layers) - 1, figsize=(4 * len(net.layers) - 1, 30))

    for frame in range(8):
        h = activations[frame]
        s = 0
        axes[frame][0].imshow(unflatten(h[0][s:s+1].squeeze(0).detach().cpu()), cmap='gray')
        for i in range(len(net.layers) - 2):
            axes[frame][i+1].imshow(unflatten_layer(h[i + 1][s:s+1].squeeze(0).detach().cpu()), cmap='gray')

    plt.show()

# %% Pruning
# activations => [frame, layer, batch_size, neurons]

# Pop the input/output layers, so the dimensions match
import copy
layer_activations = copy.deepcopy(activations)
for frame in layer_activations:
    frame.pop(0)
    frame.pop(-1)

for frame_id in range(len(layer_activations)):
    for layer in range(len(layer_activations[frame_id])):
        layer_activations[frame_id][layer] = layer_activations[frame_id][layer].clone().detach()
    layer_activations[frame_id] = torch.stack(layer_activations[frame_id])
layer_activations = torch.stack(layer_activations)

# logger.info(f"Activations Shape: {layer_activations.shape}") # [time, layer, batch_size, neurons]
sum_of_activations = layer_activations.sum(2).sum(0) # [layer, neurons]
sorted_indices = sum_of_activations.argsort(1)
important_indices = prune([net.layers[1].fw, net.layers[2].fw, net.layers[1].bw, net.layers[2].bw], sum_of_activations, args.prune_mode, args.neurons)

# %% EXP 3

import matplotlib.pyplot as plt
def mean_pairwise_cosine_similarity(vectors: torch.Tensor) -> float:
    vectors = torch.functional.F.normalize(vectors, dim=1)
    sim_matrix = vectors @ vectors.T
    n = vectors.size(0)
    return (sim_matrix.sum().item() - n) / (n * (n - 1))

def average_pearson_correlation(vectors: torch.Tensor) -> float:
    corr_matrix = torch.corrcoef(vectors.T)
    d = corr_matrix.size(0)
    return (corr_matrix.sum().item() - d) / (d * (d - 1))

def covariance_strength(vectors: torch.Tensor) -> float:
    X = vectors - vectors.mean(dim=0)
    cov = X.T @ X / (X.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return torch.norm(off_diag, p='fro').item()

# layer = 1
# wlh = 100
# # dots = [mean_pairwise_cosine_similarity(net.layers[layer].fw[sorted_indices[layer, i-wlh:i+wlh]]) for i in range(wlh, 2000 - wlh - 1)]
# dots = [net.layers[layer].fw[sorted_indices[layer, i]].dot(net.layers[layer].fw.sum() - net.layers[layer].fw[sorted_indices[layer, i]]).item() for i in range(0, 2000)]

# plt.plot(dots)
# plt.show()

# %% Prune the network

net.layers[1].fw.data = net.layers[1].fw[important_indices[0]]
net.layers[1].bw.data = net.layers[1].bw[important_indices[0]][:, important_indices[1]]
net.layers[1].fb.data = net.layers[1].fb[important_indices[0]]

net.layers[2].fw.data = net.layers[2].fw[important_indices[1]][:, important_indices[0]]
net.layers[2].bw.data = net.layers[2].bw[important_indices[1]]
net.layers[2].fb.data = net.layers[2].fb[important_indices[1]]

for layer in net.layers:
    layer.rc_features = important_indices.shape[1]

net._create_hooks_dict()

# %% Test
stats["test"] = suite.test()

# %% Strip down and save
if args.save_pruned:
    net.strip_down()
    pruned_model = suite.save(args.output, append_hash=True)
    stats["pruned_size"] = file_size(pruned_model)
    stats["pruned_model"] = pruned_model

# %% Sparsity Measurements
for type in SparsityType:
    t = str(type).split('.')[1]
    stats[t] = net.sparsity(type)

# %% Print Statistics Dictionary on STDOUT
import json

# stats['time'] = suite.time_to_train
# stats['epoch_data'] = suite.epoch_data
stats['net_stats'] = net.stats()

print(json.dumps(stats))

# %% Plot Activations
if args.plot:
    plot_activations(activations[6])

# %% Save activations
if args.save_figures:
    plot_activations(activations[6], args.save_figures)