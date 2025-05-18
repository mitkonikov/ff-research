# Code for FF Research Experiments

 > [!CAUTION]
 > Note that all code in this repository is NOT for production! It is only used for research purposes!

The code in this repository is not plug and play and needs heavy modifications
and cleanup to make it work as plug and play. Still, there are some very helpful things
one can find and use to cross-reference the results from the paper.

## Prunability

Basic outline of the files:
 - `bp.py` and `bp_load.py` are files that train and load/prune a BPNN respectively.
 - `ff_net.py` and `ff_net_load.py` are files that train and load/prune a FFNN respectively.
 - `ff_c.py` and `ff_c_load.py` are files that train and load/prune a FFNN+C respectively.
 - `ff_rnn.py` and `ff_rnn_load.py` are files that train and load/prune a FFRNN respectively.
 - `print_stats.py` uses multiple threads to spawn Python processes running the `*_load.py` scripts and gather data in .json files after the pruning process.
 - `plot_stats.py` plots the collected data using matplotlib
 - `utils.py` contains useful small utilities that are used by all `*_load.py` scripts.
 - `*.sh` are shell scripts used to run multiple training/testing configurations at once to collect data or search the space of hyperparameters.
 - There are also additional versions of the scripts with modifications for more than 2 layers.
