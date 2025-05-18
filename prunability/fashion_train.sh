python bp.py -o ./models_fashion_ff/bp.pt -d fashion -b 128 --lr 0.0001 --print-args -e 60 -v --sparsity --so ./sparsity_report_fashion/bp.json > ./models_fashion_ff/bp_42.txt
python bp_v3.py -o ./models_fashion_ff/bp3.pt -d fashion -b 128 --lr 0.0001 --print-args -e 60 -v --sparsity --so ./sparsity_report_fashion/bp3.json > ./models_fashion_ff/bp3_42.txt
python bp.py -o ./models_fashion_ff/bp120.pt -d fashion -b 128 --lr 0.0001 --print-args -e 120 -v --sparsity --so ./sparsity_report_fashion/bp.json > ./models_fashion_ff/bp_42_120e.txt
python ff_net.py -o ./models_fashion_ff/ff.pt -d fashion -b 128 --lr 0.02 --lt 20 --print-args -e 120 -v --minimize --sparsity --so ./sparsity_report_fashion/ff.json > ./models_fashion_ff/ff_42.txt
python ff_net_v3.py -o ./models_fashion_ff/ff3.pt -d fashion -b 128 --lr 0.02 --lt 20 --print-args -e 120 -v --minimize --sparsity --so ./sparsity_report_fashion/ff3.json > ./models_fashion_ff/ff3_42.txt
python ff_c.py -o ./models_fashion_ff/ffc.pt -d fashion -b 128 --lr 0.02 --lt 20 --print-args -e 60 -v --minimize > ./models_fashion_ff/ffc_42.txt
python ff_c_v3.py -o ./models_fashion_ff/ffc3.pt -d fashion -b 128 --lr 0.02 --lt 20 --print-args -e 60 -v --minimize > ./models_fashion_ff/ffc3_42.txt
python ff_rnn.py -o ./models_fashion_ff/ffrnn.pt -d fashion -b 128 --lr 0.02 --lt 20 --print-args -e 120 -v --minimize --sparsity --so ./sparsity_report_fashion/ffrnn.json > ./models_fashion_ff/ffrnn_42.txt
python ff_rnn_v3.py -o ./models_fashion_ff/ffrnn3.pt -d fashion -b 128 --lr 0.02 --lt 20 --print-args -e 120 -v --minimize --sparsity --so ./sparsity_report_fashion/ffrnn3.json > ./models_fashion_ff/ffrnn3_42.txt
