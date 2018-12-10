#!/bin/bash
docker rm $(docker ps -a -q) # remove all stopped containers
for e in RoboschoolHopper-v1.py RoboschoolAnt-v1.py RoboschoolHalfCheetah-v1.py RoboschoolHumanoid-v1.py RoboschoolReacher-v1.py RoboschoolWalker2d-v1.py
do
    docker run --name $e -d -v `pwd`:/home/user/deeprlhw1:rw -w /home/user/deeprlhw1 roboschool-cpu-vnc:latest python run_expert.py experts/$e $e-100 --num_rollouts=100
done
