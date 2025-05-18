declare -a seeds=(751257617 569547031 15741043 622633840 328441881 842099752 79254443 342403438 901220406 582768463)

for seed in "${seeds[@]}"
do
    python ff_c.py -o ./models_ff_c_max/ff_c_max.pt -s $seed -d mnist -b 128 -e 30 --lr 0.02 --lt 20 -v --print-args > ./models_ff_c_max/ff_c_$seed.txt
    python ff_c.py -o ./models_ff_c_min/ff_c_min.pt -s $seed -d mnist -b 128 -e 30 --lr 0.02 --lt 20 -v --print-args --minimize > ./models_ff_c_min/ff_c_$seed.txt
done
