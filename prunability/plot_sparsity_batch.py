"""
plot_sparsity_batch.py
----------------------
Batch plotting utility for neural network sparsity analysis.

Usage:
    python plot_sparsity_batch.py <input_dir> [--output-dir OUTPUT_DIR]

Arguments:
    input_dir:   Directory containing the sparsity report files.
    --output-dir: Directory to save the generated plots (default: ./sparsity_plots)
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Batch plotting utility for neural network sparsity analysis. "
                    "Generates and saves sparsity plots for multiple sparsity metrics (Hoyer, L1 Neg Entropy, L2 Neg Entropy, Gini) "
                    "using the results of neural network training runs. Calls plot_sparsity.py for each metric and saves the output plots "
                    "to a specified directory.")
    parser.add_argument('input_dir', type=str, help='Directory containing the sparsity report files.')
    parser.add_argument('--output-dir', type=str, default='./sparsity_plots', help='Directory to save the generated plots (default: ./sparsity_plots)')
    parser.add_argument('--plot-type', type=str, choices=['layers', 'neurons'], default='neurons',
                        help='Type of plot: "layers" for individual layers, "neurons" for multiple layer sizes (default: neurons)')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    plot_type = args.plot_type

    os.makedirs(output_dir, exist_ok=True)
    print("Plotting sparsity data...")
    print(f"Output directory: {output_dir}")

    sparsity_types = [
        ("SparsityType.HOYER", "HOYER.png"),
        ("SparsityType.L1_NEG_ENTROPY", "L1_NEG_ENTROPY.png"),
        ("SparsityType.L2_NEG_ENTROPY", "L2_NEG_ENTROPY.png"),
        ("SparsityType.GINI", "GINI.png"),
    ]

    for sparsity_type, filename in sparsity_types:
        print(f"Plotting {sparsity_type.split('.')[-1]} plot.")
        output_path = os.path.join(output_dir, filename)
        subprocess.run([
            sys.executable, "plot_sparsity.py",
            "-i", input_dir,
            "-t", sparsity_type,
            "--plot-type", plot_type,
            "--save",
            "--output", output_path
        ], check=True)

if __name__ == "__main__":
    main()
