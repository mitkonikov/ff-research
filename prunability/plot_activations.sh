# python bp_load.py -i ./models_bp/bp_mnist_3773ec.pt --save-figures ./activations_figures/bp.png
# python ff_net_load.py -i ./models_ff_min/ff_min_f93710.pt --save-figures ./activations_figures/ff.png
python ff_c_load.py -i ./models_ff_c_min/ff_c_min_a59e32.pt --save-figures ./activations_figures/ffc.png
# python ff_rnn_load.py -i ./models_ff_rnn_min/ff_rnn_min_98c25b.pt --save-figures ./activations_figures/ffrnn.png
