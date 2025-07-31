# %% Imports
from utils import ArgumentsTrain
args = ArgumentsTrain()()

import torch

from torch.optim import Adam, Optimizer

from fflib.interfaces.iff import IFF
from fflib.utils.bp_suite import BPSuite
from fflib.utils.ff_logger import logger
from fflib.utils.data.datasets import CreateDatasetFromName
from fflib.utils.maths import ComputeSparsity
from fflib.enums import SparsityType

from typing import cast

class BPDenseNet(IFF):
    def __init__(self, lr: float, in_features: int = 28 * 28, out_features: int = 10):
        super().__init__()

        layers = [
            torch.nn.Flatten(),
            torch.nn.Linear(in_features, args.neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(args.neurons, args.neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(args.neurons, out_features),
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

    def sparsity(self, sparsityType: SparsityType):
        result = {
            'layer_0': {
                'weight': ComputeSparsity(self.layers[1].weight.detach(), sparsityType).item(),
                'bias': ComputeSparsity(self.layers[1].bias.detach(), sparsityType).item()
            },
            'layer_1': {
                'weight': ComputeSparsity(self.layers[3].weight.detach(), sparsityType).item(),
                'bias': ComputeSparsity(self.layers[3].bias.detach(), sparsityType).item()
            },
            'layer_2': {
                'weight': ComputeSparsity(self.layers[5].weight.detach(), sparsityType).item(),
                'bias': ComputeSparsity(self.layers[5].bias.detach(), sparsityType).item()
            },
        }

        return result

# Setup the device
device_type = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device type: {device_type}")
device = torch.device(device_type)

logger.disabled = not args.verbose

torch.manual_seed(args.seed)

# %% Setup Dataset
logger.info("Setting up the dataset...")
dataloader = CreateDatasetFromName(
    name=args.dataset,
    batch_size=args.batch,
    validation_split=args.val_split,
    use=args.use,
)

# %% Setup Dense Backpropagation Network
net = BPDenseNet(args.lr, dataloader.get_input_shape().numel(), dataloader.get_output_shape().numel())
suite = BPSuite(net, dataloader, device)

# %% Sparsity Measurement
if args.sparsity:
    measurements = { }
    def pre_batch(net: BPDenseNet, e: int, b: int):
        if str(e) not in measurements:
            measurements[str(e)] = []
        
        result = { }
        for type in SparsityType:
            result[str(type).split('.')[1]] = net.sparsity(type)

        measurements[str(e)].append(result)

    suite.set_pre_batch_callback(pre_batch)

# %% Run Train
logger.info("Running the training procedure...")
logger.info(f"Parameters: Epochs = {args.epochs}")
suite.train(args.epochs)

# %% Run Test
logger.info("Running the testing procedure...")
suite.test()

# %% Save Model
logger.info("Saving model...")
net.strip_down()
model_path = suite.save(args.output, append_hash=True)
print(f"Model saved at {model_path}")
print(f"Time to train: {suite.time_to_train}")

# %% Save measurements
if args.sparsity:
    with open(args.so, "w") as f:
        import json
        f.write(json.dumps(measurements))
        logger.info(f"Sparsity report saved at {args.so}")