docker rm -f robo
docker run --name robo -itd -v `pwd`:/home/user/deeprlhw1:rw -p 6080:6080 roboschool-cpu-vnc:latest
