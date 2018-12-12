#!/bin/bash
docker rm $(docker ps -a -q) # remove all stopped containers
for prop in 0.1 0.3 0.5 0.7 0.9
do
    docker run --name half_cheetah-$prop -d -v `pwd`:/home/user/deeprlhw1:rw -w /home/user/deeprlhw1 roboschool-cpu-vnc:latest python behavior_cloning.py with half_cheetah use_data_prop=$prop model_dir=models/half_cheetah-$prop
done
