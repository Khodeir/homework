
b=(10000 30000 50000)
r=(0.005 0.01 0.02)

for bi in ${b[@]}
do
	for ri in ${r[@]}
	do
		echo hc_b${bi}_r${ri};
		python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b $bi -lr $ri -rtg --nn_baseline --exp_name hc_b${bi}_r${ri};

	done
done
