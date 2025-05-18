# %% Imports
from utils import ArgumentsLoad
# args = ArgumentsLoad()(["-i", "./models_bp/bp_mnist_3773ec.pt"])
# args = ArgumentsLoad()(["-i", "./models_bp/bp_29ef10.pt"])
args = ArgumentsLoad()()

import torch
import matplotlib.pyplot as plt

from torch.optim import Adam, Optimizer

from fflib.interfaces.iff import IFF
from fflib.utils.bp_suite import BPSuite
from fflib.utils.ff_logger import logger
from fflib.utils.data.mnist import NegativeGenerator as MNISTNEG
from fflib.utils.data.datasets import CreateDatasetFromName
from fflib.utils.maths import ComputeSparsity, ComputeStats
from fflib.enums import SparsityType

from utils import file_size, prune, plot_activations
from typing import cast

logger.disabled = not args.verbose

class BPDenseNet(IFF):
    def __init__(self, lr: float, in_features: int = 28 * 28, out_features: int = 10):
        super().__init__()

        layers = [
            torch.nn.Flatten(),
            torch.nn.Linear(in_features, 2000),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, 2000),
            torch.nn.ReLU(),
            torch.nn.Linear(2000, out_features),
        ]

        self.lr = lr
        self.layers = torch.nn.Sequential(*layers)
        self.criterion = torch.nn.CrossEntropyLoss()

        self._init_utils()

    def _init_utils(self) -> None:
        self.opt: Optimizer | None = Adam(self.parameters(), self.lr)

    def get_layer_count(self) -> int:
        return 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.layers(x))

    def run_train_combined(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor,
    ) -> None:
        raise NotImplementedError("Use run_train function with separate X and Y data inputs.")

    def run_train(
        self,
        x_pos: torch.Tensor,
        y_pos: torch.Tensor,
        x_neg: torch.Tensor,
        y_neg: torch.Tensor,
    ) -> None:

        if not hasattr(self, "opt") or self.opt == None:
            raise ValueError("Optimizer is not set!")

        self.opt.zero_grad()
        output = self.forward(x_pos)
        loss = self.criterion(output, y_pos)
        loss.backward()
        self.opt.step()

    def strip_down(self) -> None:
        self.opt = None


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
    validation_split=0,
    negative_generator=MNISTNEG.RANDOM
)

# %% Setup Dense Backpropagation Network
suite = BPSuite(args.input, dataloader, device)
stats["original_size"] = file_size(args.input)

# %% Run Test
if args.pretest:
    stats["pretest"] = suite.test()

# %% Get the loaded network
net = suite.net

# %% Plot weights
if args.plot:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 10))
    axes[0].imshow(net.layers[1].weight.detach().cpu(), cmap='gray')
    axes[1].imshow(net.layers[3].weight.detach().cpu(), cmap='gray')
    axes[0].text(40, 70, f'min: {net.layers[1].weight.min():.2f} max: {net.layers[1].weight.max():.2f}', bbox=dict(boxstyle='round'), fontsize=12)
    axes[1].text(40, 70, f'min: {net.layers[3].weight.min():.2f} max: {net.layers[3].weight.max():.2f}', bbox=dict(boxstyle='round'), fontsize=12)
    plt.show()

# %% Plot Histogram of Weights
if args.plot:
    plt.hist(net.layers[1].weight.flatten().detach().cpu(), bins=200)
    plt.hist(net.layers[3].weight.flatten().detach().cpu(), bins=200)
    plt.show()

# %% Plot bias
if args.plot:
    unflatten_layer = torch.nn.Unflatten(0, (50, 40))
    fig, axes = plt.subplots(1, 2, figsize=(12, 10))
    axes[0].imshow(unflatten_layer(net.layers[1].bias).detach().cpu(), cmap='gray')
    axes[1].imshow(unflatten_layer(net.layers[3].bias).detach().cpu(), cmap='gray')
    axes[0].text(1, 2, f'min: {net.layers[1].bias.min():.2f} max: {net.layers[1].bias.max():.2f}', bbox=dict(boxstyle='round'), fontsize=12)
    axes[1].text(1, 2, f'min: {net.layers[3].bias.min():.2f} max: {net.layers[3].bias.max():.2f}', bbox=dict(boxstyle='round'), fontsize=12)
    plt.show()

# %% Plot Histogram of Bias
if args.plot:
    plt.hist(net.layers[1].bias.flatten().detach().cpu(), bins=200)
    plt.hist(net.layers[3].bias.flatten().detach().cpu(), bins=200)
    plt.show()

# %% Push a batch through the network

activations = []
def forward_hook(module, input, output):
    activations.append(output)

from collections import OrderedDict

for i in range(len(net.layers)):
    if isinstance(net.layers[i], torch.nn.Linear):
        net.layers[i]._forward_hooks = OrderedDict()

hooks = []
li = []
for i in range(len(net.layers)):
    if isinstance(net.layers[i], torch.nn.Linear):
        li.append(i)
        hooks.append(net.layers[i].register_forward_hook(forward_hook))

# Get one batch
x, y = next(iter(dataloader.get_test_loader()))
x, y = x.to(device), y.to(device)

activations.insert(0, x)

# Push the batch through the network
g = net(x)

for hook in hooks:
    hook.remove()

# %% Pruning
acts: torch.Tensor = torch.stack(activations[1:-1]) # [layer, batch_size, neurons]
sum_of_activations = acts.sum(1) # [layer, neurons]
sorted_indices = sum_of_activations.argsort(1)
important_indices = prune([net.layers[1].weight, net.layers[3].weight], sum_of_activations, args.prune_mode, args.neurons)

# %% Plot activations
if args.plot:
    fig, axes = plt.subplots(1, 2, figsize=(12, 10))
    axes[0].imshow(acts[0][0].detach().reshape((50, 40)).cpu(), cmap='gray')
    axes[1].imshow(acts[1][0].detach().reshape((50, 40)).cpu(), cmap='gray')
    plt.show()
    from fflib.utils.maths import ComputeSparsity
    from fflib.enums import SparsityType
    sparsity = ComputeSparsity(acts[0][0].detach(), SparsityType.HOYER)
    print(sparsity)

# %% EXP2

# if args.plot:
    # layer = 0
    # dots = [
    #     net.layers[li[layer]].weight[sorted_indices[layer, i]].dot(
    #         net.layers[li[layer]].weight[sorted_indices[layer, i + 1]]).item() for i in range(2000 - 1)]

    # plt.plot(dots)
    # plt.show()

# %% Prune the network

net.layers[li[0]].weight.data = net.layers[li[0]].weight[important_indices[0]]
net.layers[li[1]].weight.data = net.layers[li[1]].weight[important_indices[1]][:, important_indices[0]]
net.layers[li[2]].weight.data = net.layers[li[2]].weight[:][:, important_indices[1]]
net.layers[li[0]].bias.data = net.layers[li[0]].bias[important_indices[0]]
net.layers[li[1]].bias.data = net.layers[li[1]].bias[important_indices[1]]

# %% Test
stats["test"] = suite.test()

# %% Save
if args.save_pruned:
    pruned_model = suite.save(args.output, append_hash=True)
    stats["pruned_size"] = file_size(pruned_model)
    stats["pruned_model"] = pruned_model

# %% Sparsity Measurements
def GetSparsity(net: BPDenseNet, sparsityType: SparsityType):
    result = {
        'layer_0': {
            'weight': ComputeSparsity(net.layers[li[0]].weight, sparsityType).item(),
            'bias': ComputeSparsity(net.layers[li[0]].bias, sparsityType).item()
        },
        'layer_1': {
            'weight': ComputeSparsity(net.layers[li[1]].weight, sparsityType).item(),
            'bias': ComputeSparsity(net.layers[li[1]].bias, sparsityType).item()
        },
        'layer_2': {
            'weight': ComputeSparsity(net.layers[li[2]].weight, sparsityType).item(),
            'bias': ComputeSparsity(net.layers[li[2]].bias, sparsityType).item()
        },
    }

    return result

for type in SparsityType:
    stats[str(type).split('.')[1]] = GetSparsity(net, type)

# %% Print Statistics Dictionary on STDOUT
import json

# stats['time'] = suite.time_to_train
# stats['epoch_data'] = suite.epoch_data

# for idx in [1, 3, 5]:
#     print(idx, ComputeStats(net.layers[idx].weight))

print(json.dumps(stats))

# %% Plot Activations
if args.plot:
    plot_activations(activations)

# %% Save activations
if args.save_figures:
    plot_activations(activations, args.save_figures)