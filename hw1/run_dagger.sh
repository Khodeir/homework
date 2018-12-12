rm -rf models/humanoid-dagger*
rm -rf dagger_data/*
docker rm -f humanoid-dagger && \
docker run --name humanoid-dagger -v `pwd`:/home/user/deeprlhw1:rw -w /home/user/deeprlhw1 roboschool-cpu-vnc:latest python dagger.py