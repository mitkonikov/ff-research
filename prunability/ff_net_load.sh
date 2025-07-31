declare -a arr=(1500 1000 700 500 250 100 50 25 10 5 2)

for neurons in "${arr[@]}"
do
    echo Neurons: $neurons
    # python ff_net_load.py -i ./models_ff/ff_8e5b94.pt -d mnist -b 256 -l $neurons --substract-neg | grep Test
    python bp_load.py -i ./models_bp/bp_mnist_3773ec.pt -d mnist -b 256 -l $neurons | grep Test
done
