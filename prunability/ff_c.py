# %% Imports
from utils import ArgumentsTrain
args = ArgumentsTrain()()

import torch

from fflib.nn.ff_linear import FFLinear
from fflib.nn.ffc import FFC
from fflib.interfaces.iffprobe import NoProbe
from fflib.utils.data.mnist import NegativeGenerator as MNISTNEG
from fflib.utils.data.datasets import CreateDatasetFromName
from fflib.utils.ffc_suite import FFCSuite
from fflib.utils.ff_logger import logger

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
    negative_generator=MNISTNEG.RANDOM,
    use=args.use,
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

# Setup a basic network
logger.info("Setting up FFC...")
net = FFC([layer1, layer2], classifier_lr=0.0001, output_classes=10, device=device, maximize=not args.minimize)

# %% Create Test Suite
logger.info("Setting up a TestSuite...")
suite = FFCSuite(net, NoProbe(), dataloader, device)

# %% LR Scheduler
if args.scheduler: # Enable the LR Scheduler
    def scheduler(net: FFC, e: int):
        for i in range(0, len(net.layers)):
            cur_lr = net.layers[i].get_lr()
            next_lr = min([cur_lr, cur_lr * 2 * (1 + args.epochs - e) / args.epochs])
            print(f"Layer {i} Next LR: {next_lr}")
            net.layers[i].set_lr(next_lr)

    suite.set_pre_epoch_callback(callback = scheduler)

# %% Run Train
logger.info("Running the training procedure...")
logger.info(f"Parameters: Epochs = {args.epochs}")
suite.train(args.epochs)
suite.train_switch(True)
suite.train(args.epochs)

# %% Run Test
logger.info("Running the testing procedure...")
print(f"Test Accuracy: {suite.test()}")

# %% Save Model
logger.info("Saving model...")
model_path = suite.save(args.output, append_hash=True)
print(f"Model saved at {model_path}")
