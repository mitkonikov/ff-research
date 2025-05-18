# %% Imports
from utils import ArgumentsLoad
# args = ArgumentsLoad()(['-i', './models_ff_max/ff_max_c9e323.pt'])
# args = ArgumentsLoad()(['-i', './models_ff_min/ff_min_f93710.pt'])
# args = ArgumentsLoad()(['-i', './models_ff/ff_a41d2b.pt'])
# args = ArgumentsLoad()(['-i', './models_cifar_ff/ff_79499d.pt', '-d', 'cifar10'])
args = ArgumentsLoad()()

import torch

from fflib.nn.ff_net import FFNet
from fflib.probes.one_hot import TryAllClasses
from fflib.utils.data.mnist import NegativeGenerator as MNISTNEG
from fflib.utils.data.datasets import CreateDatasetFromName
from fflib.utils.ff_suite import FFSuite
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
mnist = CreateDatasetFromName(
    name=args.dataset,
    batch_size=args.batch,
    validation_split=0,
    negative_generator=MNISTNEG.RANDOM
)

# %% Probe
logger.info("Setting up a probe...")
probe = TryAllClasses(lambda x, y: suite.net(torch.cat((x, y), 1)), output_classes=10)  # type: ignore

# %% Create Test Suite
logger.info("Setting up a TestSuite...")
suite = FFSuite(args.input, probe, mnist, device)
probe.maximize = cast(FFNet, suite.net).maximize
stats["original_size"] = file_size(args.input)

# %% Retest the model
if args.pretest:
    stats["pretest"] = suite.test()

# %% Infer
net = cast(FFNet, suite.net)

net._create_hooks_dict()

pos_activations = [torch.Tensor() for _ in range(len(net.layers) + 1)]
neg_activations = [torch.Tensor() for _ in range(len(net.layers) + 1)]
com_activations = [torch.Tensor() for _ in range(len(net.layers) + 1)]
switch = 1
def save_activation(x: torch.Tensor, id: int):
    if switch == 1:
        pos_activations[id + 1] = x.clone()
    elif switch == -1:
        neg_activations[id + 1] = x.clone()
    elif switch == 0:
        com_activations[id + 1] = x.clone()

net.register_hook('layer_activation', 'h1', save_activation)

# Take one batch
it = iter(mnist.test_loader)
batch = next(it)
x, y = batch[0].to(device), batch[1].to(device)

y_enc = mnist.encode_output(y)
input = mnist.combine_to_input(x, y_enc)
x_neg, y_neg = mnist.generate_negative(x, y, net)
input_neg = mnist.combine_to_input(x, y_neg)
input_com = torch.cat((input, input_neg), 0)

pos_activations[0] = x
neg_activations[0] = x
com_activations[0] = x
pos_activations.append(y_enc)
neg_activations.append(y_neg)

swtich = 1
net(input)
switch = -1
net(input_neg)
switch = 0
net(input_com)

# %% Plot activations
if args.plot:
    import matplotlib.pyplot as plt

    print("comb")
    unflatten = torch.nn.Unflatten(0, (28, 28))
    unflatten_layer = torch.nn.Unflatten(0, (50, 40))

    for s in range(4):
        fig, axes = plt.subplots(1, len(net.layers) + 1, figsize=(4 * len(net.layers) - 1, 5))

        hp = pos_activations
        hn = neg_activations
        axes[0].imshow(unflatten(hp[0][s:s+1].squeeze(0).detach().cpu()), cmap='gray')
        for i in range(len(net.layers)):
            act = hp[i + 1][s:s+1].squeeze(0) - hn[i + 1][s:s+1].squeeze(0)
            axes[i+1].text(0, 0, f'min: {act.flatten().min():.2f} max: {act.flatten().max():.2f}')
            axes[i+1].imshow(unflatten_layer(act.detach().cpu()), cmap='gray')

        plt.show()

# %% Plot weights
if args.plot:
    fig, axes = plt.subplots(1, 2, figsize=(12, 10))
    axes[0].imshow(net.layers[0].weight.detach().cpu(), cmap='gray')
    axes[1].imshow(net.layers[1].weight.detach().cpu(), cmap='gray')
    axes[0].text(40, 70, f'min: {net.layers[0].weight.min():.2f} max: {net.layers[0].weight.max():.2f}', bbox=dict(boxstyle='round'), fontsize=12)
    axes[1].text(40, 70, f'min: {net.layers[1].weight.min():.2f} max: {net.layers[1].weight.max():.2f}', bbox=dict(boxstyle='round'), fontsize=12)
    plt.show()

# %% Plot Histogram of Weights
if args.plot:
    plt.hist(net.layers[0].weight.flatten().detach().cpu(), bins=200, range=[-5, 5])
    plt.hist(net.layers[1].weight.flatten().detach().cpu(), bins=200, range=[-5, 5])
    plt.show()

# %% Plot bias
if args.plot:
    unflatten_layer = torch.nn.Unflatten(0, (50, 40))
    fig, axes = plt.subplots(1, 2, figsize=(12, 10))
    axes[0].imshow(unflatten_layer(net.layers[0].bias).detach().cpu(), cmap='gray')
    axes[1].imshow(unflatten_layer(net.layers[1].bias).detach().cpu(), cmap='gray')
    axes[0].text(1, 2, f'min: {net.layers[0].bias.min():.2f} max: {net.layers[0].bias.max():.2f}', bbox=dict(boxstyle='round'), fontsize=12)
    axes[1].text(1, 2, f'min: {net.layers[1].bias.min():.2f} max: {net.layers[1].bias.max():.2f}', bbox=dict(boxstyle='round'), fontsize=12)
    plt.show()

# %% Plot Histogram of Bias
if args.plot:
    plt.hist(net.layers[0].bias.flatten().detach().cpu(), bins=200)
    plt.hist(net.layers[1].bias.flatten().detach().cpu(), bins=200)
    plt.show()

# %% Pruning
pos_acts = torch.stack(pos_activations[1:-1]) # [layer, batch_size, neurons]
neg_acts = torch.stack(neg_activations[1:-1]) # [layer, batch_size, neurons]
sum_of_pos_activations = pos_acts.sum(1) # [layer, neurons]
sum_of_neg_activations = neg_acts.sum(1) # [layer, neurons]

sum_of_activations = sum_of_pos_activations - (sum_of_neg_activations if args.substract_neg else 0)
sorted_indices = sum_of_activations.argsort(1)
important_indices = prune([net.layers[0].weight, net.layers[1].weight], sum_of_activations, args.prune_mode, args.neurons)

# %% CORRELATION EXPERIMENT:
# dots1 = []
# for i in range(20):
#     last = sorted_indices[0, -128-i*16-1:-i*16-1]
#     d1 = 0
#     for i in range(128):
#         for j in range(128):
#             d1 += net.layers[0].weight[last[i]].dot(net.layers[0].weight[last[j]])
#     dots1.append(d1.item())

# dots2 = []
# for i in range(20):
#     perm = torch.randperm(2000)[:128]
#     last2 = sorted_indices[0, perm]
#     d2 = 0
#     for i in range(128):
#         for j in range(128):
#             d2 += net.layers[0].weight[last2[i]].dot(net.layers[0].weight[last2[j]])
#     dots2.append(d2.item())

# print(dots1)
# print(torch.Tensor(dots1).mean().item())
# print(torch.Tensor(dots1).std().item())
# print(torch.Tensor(dots1).max().item())
# print()
# print(dots2)
# print(torch.Tensor(dots2).mean().item())
# print(torch.Tensor(dots2).std().item())
# print(torch.Tensor(dots2).max().item())

# %% EXP2

# import matplotlib.pyplot as plt
# layer = 1
# window_size = 100
# window = torch.hamming_window(window_size, device=device)
# window = window / window.norm(1)

# n = net.layers[layer].weight.shape[0]
# dots = []
# for i in range(n):
#     curr = net.layers[layer].weight[sorted_indices[layer, window_size//2]].detach()
#     for j in range(1, window_size//2 - 1):
#         if i+j < n:
#             curr *= net.layers[layer].weight[sorted_indices[layer, i+j]].detach() * window[window_size//2+j]
#         if i-j > 0:
#             curr *= net.layers[layer].weight[sorted_indices[layer, i-j]].detach() * window[window_size//2-j]
#     dots.append(curr.sum().item())

#     #   Corr(i, i-2) * window[M-2]
#     # + Corr(i, i-1) * window[M-1]
#     # + Corr(i, i+1) * window[M+1]
#     # + Corr(i, i+2) * window[M+2]

# # dots = [
# #     net.layers[layer].weight[sorted_indices[layer, i]].dot(
# #         net.layers[layer].weight[sorted_indices[layer, i + 1]]).item() for i in range(2000 - 1)]

# plt.plot(dots)
# plt.show()

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

# layer = 0
# # dots = [mean_pairwise_cosine_similarity(net.layers[layer].weight[sorted_indices[layer, i-50:i+50]]) for i in range(50, 2000 - 51)]
# dots = [net.layers[layer].weight[sorted_indices[layer, i]].dot(net.layers[layer].weight.sum(0) - net.layers[layer].weight[sorted_indices[layer, i]]).item() for i in range(0, 2000)]

# plt.plot(dots)
# plt.show()

# %% Prune the network

net.layers[0].weight.data = net.layers[0].weight[important_indices[0]]
net.layers[0].bias.data = net.layers[0].bias[important_indices[0]]

net.layers[1].weight.data = net.layers[1].weight[important_indices[1]][:, important_indices[0]]
net.layers[1].bias.data = net.layers[1].bias[important_indices[1]]

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
    plot_activations(pos_activations)

# %% Save activations
if args.save_figures:
    plot_activations(pos_activations, args.save_figures)