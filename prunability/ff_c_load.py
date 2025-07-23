# %% Imports
from utils import ArgumentsLoad
# args = ArgumentsLoad()(['-i', './models_ff_c_max/ff_c_max_764146.pt'])
args = ArgumentsLoad()()

import torch

from fflib.nn.ffc import FFC
from fflib.interfaces.iffprobe import NoProbe
from fflib.utils.data.mnist import NegativeGenerator as MNISTNEG
from fflib.utils.data.datasets import CreateDatasetFromName
from fflib.utils.ffc_suite import FFCSuite
from fflib.enums import SparsityType
from fflib.utils.ff_logger import logger

from utils import file_size, prune, plot_activations, plot_activations_hsv

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
    validation_split=0,
    negative_generator=MNISTNEG.RANDOM
)

# %% Create Test Suite
logger.info("Setting up a TestSuite...")
suite = FFCSuite(args.input, NoProbe(), dataloader, device)
stats["original_size"] = file_size(args.input)

# %% Retest the model
if args.pretest:
    stats["pretest"] = suite.test()

# %% Infer
from typing import cast

net = cast(FFC, suite.net)

activations = [torch.Tensor() for _ in range(len(net.layers) + 1)]
def save_activation(x: torch.Tensor, id: int):
    activations[id + 1] = x.clone()

net.register_hook('layer_activation', 'h1', save_activation)

# Take one batch
it = iter(dataloader.test_loader)
batch = next(it)
x, y = batch[0].to(device), batch[1].to(device)

y = dataloader.encode_output(y)
input = dataloader.combine_to_input(x, y)

activations[0] = x
activations.append(y)

net(input)

net._create_hooks_dict()

# %% Plot weights
if args.plot:
    import matplotlib.pyplot as plt

    unflatten = torch.nn.Unflatten(0, (28, 28))
    unflatten_layer = torch.nn.Unflatten(0, (50, 40))

    for s in range(4):
        fig, axes = plt.subplots(1, len(net.layers) + 1, figsize=(4 * len(net.layers) - 1, 5))

        h = activations
        axes[0].imshow(unflatten(h[0][s:s+1].squeeze(0).detach().cpu()), cmap='gray')
        for i in range(len(net.layers)):
            axes[i+1].imshow(unflatten_layer(h[i + 1][s:s+1].squeeze(0).detach().cpu()), cmap='gray')

        plt.show()

# %% Pruning

layer_activations = torch.stack(activations[1:-1]) # [layer, batch_size, neurons]
sum_of_activations = layer_activations.sum(1) # [layer, neurons]
sorted_indices = sum_of_activations.argsort(1)
important_indices = prune([net.layers[0].weight, net.layers[1].weight], sum_of_activations, args.prune_mode, args.neurons)

# %% Prune the network

net.layers[0].weight.data = net.layers[0].weight[important_indices[0]]
net.layers[1].weight.data = net.layers[1].weight[important_indices[1]][:, important_indices[0]]
net.layers[0].bias.data = net.layers[0].bias[important_indices[0]]
net.layers[1].bias.data = net.layers[1].bias[important_indices[1]]

classifier_important_indices = torch.cat((important_indices[0], important_indices[1] + 2000))
net.classifier.weight.data = net.classifier.weight[:, classifier_important_indices]

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
stats['net_stats'] = net.stats()
print(json.dumps(stats))

# %% Plot Activations
if args.plot:
    plot_activations(activations)

# %% Save activations
if args.save_figures:
    plot_activations(activations, args.save_figures)

# %% Save activations in HSV style
if args.save_activations_hsv:
    net = cast(FFC, suite.net)

    net._create_hooks_dict()

    # activations[class, layer, neuron]
    activations = [
        [torch.zeros(net.layers[1].out_features).to(device) for layer in range(2)]
        for _ in range(10)
    ]
    
    def save_activation(x: torch.Tensor, layer_idx: int):
        # x: [batch_size, num_neurons]
        for i in range(x.size(0)):
            label = current_labels[i].item()
            activations[label][layer_idx] += x[i].detach().abs()

    net.register_hook('layer_activation', 'h1', save_activation)

    it = iter(dataloader.test_loader)
    for i in range(10):
        batch = next(it)
        x, y = batch[0].to(device), batch[1].to(device)

        current_labels = y

        y_enc = dataloader.encode_output(y)
        y_enc = torch.zeros_like(y_enc)
        input = dataloader.combine_to_input(x, y_enc)

        net(input)

    plot_activations_hsv(activations, args.save_activations_hsv)