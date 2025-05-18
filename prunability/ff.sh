python ff_net.py -o ./models_ff/ff.pt -e 60 --scheduler --lr 0.02 --lt 20 --print-args > ./models_ff/ff_01.txt
python ff_net.py -o ./models_ff/ff.pt -e 60 --scheduler --lr 0.01 --lt 20 --print-args > ./models_ff/ff_02.txt
python ff_net.py -o ./models_ff/ff.pt -e 60 --scheduler --lr 0.006 --lt 20 --print-args > ./models_ff/ff_03.txt
python ff_net.py -o ./models_ff/ff.pt -e 60 --scheduler --lr 0.003 --lt 20 --print-args > ./models_ff/ff_04.txt
python ff_net.py -o ./models_ff/ff.pt -e 60 --scheduler --lr 0.01 --lt 5 --print-args > ./models_ff/ff_05.txt
python ff_net.py -o ./models_ff/ff.pt -e 60 --scheduler --lr 0.003 --lt 1 --print-args > ./models_ff/ff_06.txt

python ff_c.py -o ./models_ffc/ffc.pt -e 30 --scheduler --lr 0.02 --lt 20 --print-args > ./models_ffc/ffc_01.txt
python ff_c.py -o ./models_ffc/ffc.pt -e 30 --scheduler --lr 0.01 --lt 20 --print-args > ./models_ffc/ffc_02.txt
python ff_c.py -o ./models_ffc/ffc.pt -e 30 --scheduler --lr 0.006 --lt 20 --print-args > ./models_ffc/ffc_03.txt
python ff_c.py -o ./models_ffc/ffc.pt -e 30 --scheduler --lr 0.003 --lt 20 --print-args > ./models_ffc/ffc_04.txt
python ff_c.py -o ./models_ffc/ffc.pt -e 30 --scheduler --lr 0.01 --lt 5 --print-args > ./models_ffc/ffc_05.txt
python ff_c.py -o ./models_ffc/ffc.pt -e 30 --scheduler --lr 0.003 --lt 1 --print-args > ./models_ffc/ffc_06.txt
