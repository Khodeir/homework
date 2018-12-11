FROM ikeyasu/roboschool:latest

USER root
RUN pip3.5 uninstall -y tensorflow-gpu
RUN pip3.5 install tensorflow
RUN ln -s /usr/bin/python3.5 /usr/local/bin/python
RUN pip3.5 install sacred
RUN pip3.5 install seaborn
ENV APP "lxterminal -e bash"
