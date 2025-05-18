import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from argparse import ArgumentParser
from typing import List
from abc import abstractmethod
from math import ceil

def file_size(file: str):
    """Returns the file size in MB"""
    return os.stat(file).st_size / (1024 * 1024)

class ArgumentParserBase:
    def __init__(self):
        return
    
    @abstractmethod
    def create(self) -> ArgumentParser:
        pass

    @abstractmethod
    def setup(self, parser: ArgumentParser):
        pass

    @abstractmethod
    def run(self, parser: argparse.ArgumentParser, args_inject: List[str] | None = None):
        pass

    def __call__(self, args_inject: List[str] | None = None) -> ArgumentParser:
        parser = self.create()
        self.setup(parser)
        return self.run(parser, args_inject)
    

class ArgumentsTrain(ArgumentParserBase):
    def create(self) -> ArgumentParser:
        formatter_class = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=300)
        parser = ArgumentParser(exit_on_error=False, formatter_class=formatter_class)
        return parser

    def setup(self, parser: argparse.ArgumentParser):
        parser.add_argument("-s", "--seed", help="Set seed", default=42, type=int)
        parser.add_argument("-d", "--dataset", help="Dataset", default="MNIST", type=str)
        parser.add_argument("-b", "--batch", help="Batch size", default=128, type=int)
        parser.add_argument("-o", "--output", help="Output path", default="./models/network.pt", type=str)
        parser.add_argument("-v", "--verbose", help="Verbose", action="store_true")
        parser.add_argument("-e", "--epochs", help="Number of epochs", default=60, type=int)
        parser.add_argument("-u", "--use", help="Part of MNIST to use", default=1.0, type=float)
        parser.add_argument("--val-split", help="Validation Split", default=0.1, type=float)
        parser.add_argument("--scheduler", help="Enable the Learning Rate Scheduler", action="store_true")
        parser.add_argument("--minimize", help="Optimize by minimization of the goodness", action="store_true")
        parser.add_argument("--print-args", help="Print the CLI arguments", action="store_true")
        parser.add_argument("--lr", help="Learning Rate", default=0.005, type=float)
        parser.add_argument("--lt", help="Loss Threshold", default=20, type=float)
        parser.add_argument("--sparsity", help="Save sparsity report for the training phase", action="store_true")
        parser.add_argument("--so", help="Output path for the sparsity report", default="./sparsity_report.json", type=str)

    def run(self, parser: argparse.ArgumentParser, args_inject: List[str] | None = None):
        # It should skip the argument parser if it finds it runs within an IPython shell.
        args = parser.parse_args(args=(args_inject if args_inject is not None else None))
        
        if args.print_args:
            print(' '.join(sys.argv))
            print()

        return args

class ArgumentsLoad(ArgumentParserBase):
    def create(self) -> ArgumentParser:
        formatter_class = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=300)
        parser = ArgumentParser(exit_on_error=False, formatter_class=formatter_class)
        return parser

    def setup(self, parser: ArgumentParser):
        DEFAULT_NET = "./models/ff_net_mnist_E60_1b7500.pt"
        parser.add_argument("-s", "--seed", help="Set seed", default=42, type=int)
        parser.add_argument("-d", "--dataset", help="Dataset", default="MNIST", type=str)
        parser.add_argument("-b", "--batch", help="Batch size", default=128, type=int)
        parser.add_argument("-i", "--input", help="Path to the network", default=DEFAULT_NET, type=str)
        parser.add_argument("--pretest", help="Run a test after loading the network", action="store_true")
        parser.add_argument("--plot", help="Plot the activations", action="store_true")
        parser.add_argument("-n", "--neurons", help="Leave the most N active neurons", default=500, type=int)
        parser.add_argument("--save-pruned", help="Save the pruned model", action="store_true")
        parser.add_argument("-o", "--output", help="Output path", default="./models/pruned.pt", type=str)
        parser.add_argument("-v", "--verbose", help="Verbose", action="store_true")
        parser.add_argument("--lr", help="Learning Rate", default=0.005, type=float)
        parser.add_argument("--lt", help="Loss Threshold", default=20, type=float)
        parser.add_argument("--print-args", help="Print the CLI arguments", action="store_true")
        parser.add_argument("--substract-neg", help="Substract the negative activations", action="store_true")
        parser.add_argument("--prune-mode", help="Pruning mode [random, first, last]", default="random", type=str)
        parser.add_argument("--save-figures", help="Path where to save figures. It will save only if provided.", default=None, type=str)
    
    def run(self, parser: argparse.ArgumentParser, args_inject: List[str] | None = None):
        # It should skip the argument parser if it finds it runs within an IPython shell.
        args = parser.parse_args(args=(["--plot"] + args_inject if args_inject is not None else None))

        if args.print_args:
            print(' '.join(sys.argv))
            print()

        return args


def prune(layer_weights: List[torch.Tensor], activations: torch.Tensor, prune_mode: str, leave_neurons: int) -> torch.Tensor:
    # activations.shape -> [layer, neurons]
    initial_neurons = activations.shape[1]
    sorted_indices = activations.argsort(1)

    if prune_mode == "first":
        return sorted_indices[:, :leave_neurons]
    elif prune_mode == "last":
        return sorted_indices[:, -leave_neurons:]
    elif prune_mode == "ends":
        r1 = sorted_indices[:, :leave_neurons//2]
        r2 = sorted_indices[:, -leave_neurons//2:]
        return torch.concat((r1, r2), 1)
    elif prune_mode == "invert-ends":
        inv = initial_neurons - leave_neurons
        return sorted_indices[:, inv//2:-inv//2-1]
    elif prune_mode == "random":
        important_indices1 = torch.randperm(initial_neurons)[:leave_neurons]
        important_indices2 = torch.randperm(initial_neurons)[:leave_neurons]
        return torch.stack((important_indices1, important_indices2))
    elif prune_mode == "sum":
        idx1 = layer_weights[0].sum(1).argsort(0, descending=True)[:leave_neurons]
        idx2 = layer_weights[1].sum(1).argsort(0, descending=True)[:leave_neurons]
        return torch.stack((idx1, idx2))
    elif prune_mode == "random-weights":
        with torch.no_grad():
            torch.nn.functional.dropout(layer_weights[0], 1.00 - leave_neurons / initial_neurons, True, True)
            torch.nn.functional.dropout(layer_weights[1], 1.00 - leave_neurons / initial_neurons, True, True)
        important_indices = torch.arange(0, initial_neurons, 1)
        return torch.stack((important_indices, important_indices))
    elif prune_mode == "threshold-weights":
        with torch.no_grad():
            for i in range(2):
                all_weights = layer_weights[i].abs().flatten()
                k = int(leave_neurons / initial_neurons * all_weights.numel())
                if k < 1:
                    threshold = float('inf')  # prune everything
                else:
                    threshold = torch.kthvalue(all_weights, all_weights.numel() - k + 1).values.item()

                layer_weights[i].masked_fill_(layer_weights[i].abs() < threshold, 0)

        important_indices = torch.arange(0, initial_neurons, 1)
        return torch.stack((important_indices, important_indices))
    else:
        raise RuntimeError("Invalid pruning mode!")


def plot_activations(activations, output: str | None = None):
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.1], wspace=0)

    for i, activation in enumerate(activations):
        activation = activation.detach().cpu()[0]
        num_units = activation.shape[0]
        num_cols = min(40, num_units)  # Display up to 10 activations
        num_rows = int(ceil(num_units / num_cols))
        axes = fig.add_subplot(gs[i])
        axes.axis('off')
        aspect = None
        if i == 0:
            activation = activation.reshape(28, 28)
            axes.title.set_text('Input image')
        else:
            activation = activation.reshape(num_rows, num_cols)
            axes.title.set_text(f'Layer {i}')
            if i == len(activations) - 1:
                activation = activation.reshape(10, 1)
                axes.title.set_text('Output')
                ticks = list(range(0, num_units))
                axes.set_xticks([], [])
                axes.set_yticks(ticks, ticks)
                axes.axis('on')
                axes.yaxis.tick_right()
                aspect = 'auto'

        axes.imshow(activation, cmap='gray', aspect=aspect)

    if output is not None:
        plt.savefig(output)
    else:
        plt.tight_layout()
        plt.show()