rm -rf models/humanoid-dagger*
rm -rf dagger_data/*
docker rm -f humanoid-dagger
mkdir dagger_data
docker run -d --name humanoid-dagger -v `pwd`:/home/user/deeprlhw1:rw -w /home/user/deeprlhw1 roboschool-cpu-vnc:latest python dagger.py

docker wait humanoid-dagger

docker run -d --name humanoid-dagger -v `pwd`:/home/user/deeprlhw1:rw -w /home/user/deeprlhw1 roboschool-cpu-vnc:latest python generate_dagger_comparison_plot.py