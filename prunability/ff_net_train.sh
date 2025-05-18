declare -a seeds=(751257617 569547031 15741043 622633840 328441881 842099752 79254443 342403438 901220406 582768463)

for seed in "${seeds[@]}"
do
    python ff_net.py -o ./models_ff_max/ff_max.pt -s $seed -d mnist -b 128 -e 60 --lr 0.02 --lt 20 -v --print-args > ./models_ff_max/ff_$seed.txt
    python ff_net.py -o ./models_ff_min/ff_min.pt -s $seed -d mnist -b 128 -e 60 --lr 0.02 --lt 20 -v --print-args --minimize > ./models_ff_min/ff_$seed.txt
done

# 3 LAYERS
# for seed in "${seeds[@]}"
# do
#     python ff_net_v3.py -o ./models_ff_v3_max/ff_max.pt -s $seed -d mnist -b 128 -e 60 --lr 0.02 --lt 20 -v --print-args > ./models_ff_v3_max/ff_$seed.txt
#     python ff_net_v3.py -o ./models_ff_v3_min/ff_min.pt -s $seed -d mnist -b 128 -e 60 --lr 0.02 --lt 20 -v --print-args --minimize > ./models_ff_v3_min/ff_$seed.txt
# done
