python bp.py -o ./models_cifar_ff/bp.pt -d cifar10 -b 128 --lr 0.0001 --print-args -e 60 -v --sparsity --so ./sparsity_report_cifar/bp.json > ./models_cifar_ff/bp_42.txt
python bp_v3.py -o ./models_cifar_ff/bp3.pt -d cifar10 -b 128 --lr 0.0001 --print-args -e 60 -v --sparsity --so ./sparsity_report_cifar/bp3.json > ./models_cifar_ff/bp3_42.txt
python bp.py -o ./models_cifar_ff/bp120.pt -d cifar10 -b 128 --lr 0.0001 --print-args -e 120 -v --sparsity --so ./sparsity_report_cifar/bp.json > ./models_cifar_ff/bp_42_120e.txt
python ff_net.py -o ./models_cifar_ff/ff.pt -d cifar10 -b 128 --lr 0.02 --lt 20 --print-args -e 120 -v --minimize --sparsity --so ./sparsity_report_cifar/ff.json > ./models_cifar_ff/ff_42.txt
python ff_net_v3.py -o ./models_cifar_ff/ff3.pt -d cifar10 -b 128 --lr 0.02 --lt 20 --print-args -e 120 -v --minimize --sparsity --so ./sparsity_report_cifar/ff3.json > ./models_cifar_ff/ff3_42.txt
python ff_c.py -o ./models_cifar_ff/ffc.pt -d cifar10 -b 128 --lr 0.02 --lt 20 --print-args -e 60 -v --minimize > ./models_cifar_ff/ffc_42.txt
python ff_c_v3.py -o ./models_cifar_ff/ffc3.pt -d cifar10 -b 128 --lr 0.02 --lt 20 --print-args -e 60 -v --minimize > ./models_cifar_ff/ffc3_42.txt
python ff_rnn.py -o ./models_cifar_ff/ffrnn.pt -d cifar10 -b 128 --lr 0.02 --lt 20 --print-args -e 120 -v --minimize --sparsity --so ./sparsity_report_cifar/ffrnn.json > ./models_cifar_ff/ffrnn_42.txt
python ff_rnn_v3.py -o ./models_cifar_ff/ffrnn3.pt -d cifar10 -b 128 --lr 0.02 --lt 20 --print-args -e 120 -v --minimize --sparsity --so ./sparsity_report_cifar/ffrnn3.json > ./models_cifar_ff/ffrnn3_42.txt
