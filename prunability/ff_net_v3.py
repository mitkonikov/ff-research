# %% Imports
from utils import ArgumentsTrain
args = ArgumentsTrain()()

import torch

from fflib.nn.ff_linear import FFLinear
from fflib.nn.ff_net import FFNet
from fflib.probes.one_hot import TryAllClasses
from fflib.utils.data.mnist import NegativeGenerator as MNISTNEG
from fflib.utils.ff_suite import FFSuite
from fflib.utils.ff_logger import logger
from fflib.utils.data.datasets import CreateDatasetFromName
from fflib.enums import SparsityType

# Setup the device
device_type = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device type: {device_type}")
device = torch.device(device_type)

torch.manual_seed(args.seed)

# %% Setup Dataset
logger.info("Setting up the dataset...")
dataloader = CreateDatasetFromName(
    name=args.dataset,
    batch_size=args.batch,
    validation_split=args.val_split,
    use=args.use,
    negative_generator=MNISTNEG.RANDOM
)

# %% Setup the layers
logger.info("Setting up layers...")
lt = args.lt
lr = args.lr

layer1 = FFLinear(
    in_features=
        dataloader.get_input_shape().numel() +
        dataloader.get_output_shape().numel(),
    out_features=2000,
    loss_threshold=lt,
    lr=lr,
    device=device,
)

layer2 = FFLinear(
    in_features=2000,
    out_features=2000,
    loss_threshold=lt,
    lr=lr,
    device=device,
)

layer3 = FFLinear(
    in_features=2000,
    out_features=2000,
    loss_threshold=lt,
    lr=lr,
    device=device,
)

# Setup a basic network
logger.info("Setting up FFNet...")
net = FFNet([layer1, layer2, layer3], device, maximize=not args.minimize)

# %% Probe
logger.info("Setting up a probe...")
probe = TryAllClasses(lambda x, y: net(torch.cat((x, y), 1)), output_classes=10, maximize=not args.minimize)

# %% Create Test Suite
logger.info("Setting up a TestSuite...")
suite = FFSuite(net, probe, dataloader, device)

# %% LR Scheduler
if args.scheduler: # Enable the LR Scheduler
    def scheduler(net: FFNet, e: int):
        for i in range(0, len(net.layers)):
            cur_lr = net.layers[i].get_lr()
            next_lr = min([cur_lr, cur_lr * 2 * (1 + args.epochs - e) / args.epochs])
            print(f"Layer {i} Next LR: {next_lr}")
            net.layers[i].set_lr(next_lr)

    suite.set_pre_epoch_callback(callback = scheduler)

# %% Sparsity Measurement
if args.sparsity:
    measurements = { }
    def pre_batch(net: FFNet, e: int, b: int):
        if str(e) not in measurements:
            measurements[str(e)] = []

        result = { }
        for type in SparsityType:
            result[str(type).split('.')[1]] = net.sparsity(type)

        result['HISTOGRAM'] = {
            'layer_0': {
                'weight': net.layers[0].weight.detach().flatten().cpu().histogram(200).hist.tolist(),
                'bias': net.layers[0].bias.detach().flatten().cpu().histogram(200).hist.tolist(),
            },
            'layer_1': {
                'weight': net.layers[1].weight.detach().flatten().cpu().histogram(200).hist.tolist(),
                'bias': net.layers[1].bias.detach().flatten().cpu().histogram(200).hist.tolist(),
            },
            'layer_2': {
                'weight': net.layers[2].weight.detach().flatten().cpu().histogram(200).hist.tolist(),
                'bias': net.layers[2].bias.detach().flatten().cpu().histogram(200).hist.tolist(),
            },
        }

        measurements[str(e)].append(result)

    suite.set_pre_batch_callback(pre_batch)

# %% Run Train
logger.info("Running the training procedure...")
suite.train(args.epochs)

# %% Run Test
logger.info(f"Running the testing procedure...")
print(f"Test Accuracy: {suite.test()}")

# %% Save model
logger.info("Saving model...")
model_path = suite.save(args.output, append_hash=True)
print(f"Model saved at {model_path}")

# %% Save measurements
if args.sparsity:
    with open(args.so, "w") as f:
        import json
        f.write(json.dumps(measurements))