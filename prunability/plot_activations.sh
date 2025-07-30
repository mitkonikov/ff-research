declare output_dir="./activations_figures"

if [ -d "$output_dir" ]; then
    echo "Output directory already exists. Are you sure to overwrite it? [y/N]"
    read -r answer
    if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
        echo "Exiting without overwriting."
        exit 1
    fi
else
    mkdir -p $output_dir
    echo "Created output directory: $output_dir"
fi

if [ "$1" == "--single-sample" ]; then
    echo "Using old visualizations..."
    python bp_load.py -i ./models_bp/bp_mnist_3773ec.pt --save-figures $output_dir/bp.png
    python ff_net_load.py -i ./models_ff_min/ff_min_f93710.pt --save-figures $output_dir/ff.png
    python ff_c_load.py -i ./models_ff_c_min/ff_c_min_a59e32.pt --save-figures $output_dir/ffc.png
    python ff_rnn_load.py -i ./models_ff_rnn_min/ff_rnn_min_98c25b.pt --save-figures $output_dir/ffrnn.png
else
    python bp_load.py -i ./models_bp/bp_mnist_3773ec.pt -d mnist -b 256 -n 2000 --save-activations-hsv $output_dir/bp.png
    python ff_net_load.py -i ./models_ff_min/ff_min_f93710.pt -d mnist -b 256 -n 2000 --save-activations-hsv $output_dir/ff.png
    python ff_c_load.py -i ./models_ff_c_min/ff_c_min_a59e32.pt -d mnist -b 256 -n 2000 --save-activations-hsv $output_dir/ffc.png
    python ff_rnn_load.py -i ./models_ff_rnn_min/ff_rnn_min_98c25b.pt -d mnist -b 256 -n 2000 --save-activations-hsv $output_dir/ffrnn.png
fi
