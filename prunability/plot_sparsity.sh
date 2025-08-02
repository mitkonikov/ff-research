# Script to plot sparsity data from the training process

if [ -z "$2" ]; then
    output_dir="./sparsity_plots"
else
    output_dir="$2"
fi

mkdir -p $output_dir
echo "Plotting sparsity data..."
echo "Output directory: $output_dir"

input_dir="$1"

echo "Plotting HOYER plot."
python plot_sparsity.py -i $input_dir -t SparsityType.HOYER --save --output $output_dir/HOYER.png

echo "Plotting L1 plot."
python plot_sparsity.py -i $input_dir -t SparsityType.L1_NEG_ENTROPY --save --output $output_dir/L1_NEG_ENTROPY.png

echo "Plotting L2 plot."
python plot_sparsity.py -i $input_dir -t SparsityType.L2_NEG_ENTROPY --save --output $output_dir/L2_NEG_ENTROPY.png

echo "Plotting GINI plot."
python plot_sparsity.py -i $input_dir -t SparsityType.GINI --save --output $output_dir/GINI.png
