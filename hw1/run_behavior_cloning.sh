#!/bin/bash
docker rm $(docker ps -a -q) # remove all stopped containers
var=""
for e in half_cheetah ant hopper humanoid reacher walker
do
    docker run --name $e -d -v `pwd`:/home/user/deeprlhw1:rw -w /home/user/deeprlhw1 roboschool-cpu-vnc:latest python behavior_cloning.py with $e
	var=$var\ $e
done
docker wait $var
