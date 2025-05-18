# %% Imports
from utils import ArgumentsTrain
args = ArgumentsTrain()()

import torch

from fflib.utils.data.mnist import NegativeGenerator as MNISTNEG
from fflib.utils.data.datasets import CreateDatasetFromName
from fflib.nn.ff_rnn import FFRNN
from fflib.probes.one_hot import TryAllClasses
from fflib.utils.ffrnn_suite import FFRNNSuite
from fflib.utils.ff_logger import logger
from fflib.enums import SparsityType

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
    negative_generator=MNISTNEG.RANDOM,
    use=args.use,
)

# %% Setup the network
logger.info("Setting up the FFRNN...")

net = FFRNN.from_dimensions(
    dimensions=[dataloader.get_input_shape().numel(), 2000, 2000, 2000, dataloader.get_output_shape().numel()],
    K_train=10,
    K_testlow=3,
    K_testhigh=8,
    maximize=not args.minimize,
    activation_fn=torch.nn.ReLU(),
    loss_threshold=args.lt,
    optimizer=torch.optim.Adam,
    lr=args.lr,
    beta=0.7,
    device=device,
)

# %% Probe
logger.info("Setting up a probe...")
probe = TryAllClasses(lambda x, y: net(x, y), output_classes=10, maximize=not args.minimize)

# %% Create Test Suite
logger.info("Setting up a TestSuite...")
suite = FFRNNSuite(net, probe, dataloader, device)

# %% LR Scheduler
if args.scheduler: # Enable the LR Scheduler
    def scheduler(net: FFRNN, e: int):
        for i in range(1, len(net.layers) - 1):
            cur_lr = net.layers[i].get_lr()
            next_lr = min([cur_lr, cur_lr * 2 * (1 + args.epochs - e) / args.epochs])
            print(f"Layer {i} Next LR: {next_lr}")
            net.layers[i].set_lr(next_lr)

    suite.set_pre_epoch_callback(callback = scheduler)

# %% Sparsity Measurement
if args.sparsity:
    measurements = { }
    def pre_batch(net: FFRNN, e: int, b: int):
        if str(e) not in measurements:
            measurements[str(e)] = []

        result = { }
        for type in SparsityType:
            result[str(type).split('.')[1]] = net.sparsity(type)

        result['HISTOGRAM'] = {
            'layer_0': {
                'fw': net.layers[1].fw.detach().flatten().cpu().histogram(200).hist.tolist(),
                'bw': net.layers[1].bw.detach().flatten().cpu().histogram(200).hist.tolist(),
                'bias': net.layers[1].fb.detach().flatten().cpu().histogram(200).hist.tolist(),
            },
            'layer_1': {
                'fw': net.layers[2].fw.detach().flatten().cpu().histogram(200).hist.tolist(),
                'bw': net.layers[2].bw.detach().flatten().cpu().histogram(200).hist.tolist(),
                'bias': net.layers[2].fb.detach().flatten().cpu().histogram(200).hist.tolist(),
            },
            'layer_2': {
                'fw': net.layers[3].fw.detach().flatten().cpu().histogram(200).hist.tolist(),
                'bw': net.layers[3].bw.detach().flatten().cpu().histogram(200).hist.tolist(),
                'bias': net.layers[3].fb.detach().flatten().cpu().histogram(200).hist.tolist(),
            },
        }

        measurements[str(e)].append(result)

    suite.set_pre_batch_callback(pre_batch)

# %% Run Train
logger.info("Running the training procedure...")
logger.info(f"Parameters: Epochs = {args.epochs}")
suite.train(args.epochs)

# %% Run Test
logger.info("Running the testing procedure...")
print(f"Test Accuracy: {suite.test()}")

# %% Save Model
logger.info("Saving model...")
model_path = suite.save(args.output, append_hash=True)
print(f"Model saved at {model_path}")

# %% Save measurements
if args.sparsity:
    with open(args.so, "w") as f:
        import json
        f.write(json.dumps(measurements))