#!/bin/bash
docker rm $(docker ps -a -q) # remove all stopped containers
name=half_cheetah
for prop in 0.1 0.3 0.5 0.7 0.9
do
    docker run --name $name-$prop -d -v `pwd`:/home/user/deeprlhw1:rw -w /home/user/deeprlhw1 roboschool-cpu-vnc:latest python behavior_cloning.py with $name use_data_prop=$prop model_dir=models/$name-$prop
	var=$var\ $name-$prop
done

docker wait $var
docker run --user user -v `pwd`:/home/user/deeprlhw1:rw -w /home/user/deeprlhw1 roboschool-cpu-vnc:latest python generate_box_and_whisker.py