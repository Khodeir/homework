python train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 -n 100 -e 3 -l 2 -lc 4 -s 32 -sc 64 -lrc 0.01 -b 30000 -lr 0.02 --exp_name ac3_lrc0.01_sc64_lc4 -ntu 10 -ngsptu 10 &
python train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 -n 100 -e 3 -l 2 -lc 4 -s 32 -sc 64 -lrc 0.02 -b 30000 -lr 0.02 --exp_name ac3_lrc0.02_sc64_lc4 -ntu 10 -ngsptu 10

wait

python train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 -n 100 -e 3 -l 2 -lc 8 -s 32 -sc 64 -lrc 0.01 -b 30000 -lr 0.02 --exp_name ac3_lrc0.01_sc64_lc8 -ntu 10 -ngsptu 10 &
python train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 -n 100 -e 3 -l 2 -lc 8 -s 32 -sc 64 -lrc 0.02 -b 30000 -lr 0.02 --exp_name ac3_lrc0.02_sc64_lc8 -ntu 10 -ngsptu 10
