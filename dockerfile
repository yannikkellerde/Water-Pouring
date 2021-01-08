FROM nvidia/cuda:10.0-base-ubuntu18.04
RUN apt-get update

# All that stuff is needed to get the simulator going
RUN apt-get install -y git curl python3.7 cmake python3-setuptools \
    build-essential libx11-dev xorg-dev libglu1-mesa-dev python3.7-dev

# Get pip for python 3.7
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.7 get-pip.py

RUN mkdir /usr/src/bachelorthesis
WORKDIR /usr/src/bachelorthesis
# Get all code and requirements ready
COPY requirements.txt .
COPY water-pouring ./water-pouring
RUN git clone git://github.com/yannikkellerde/SPlisHSPlasH.git
RUN git clone git://github.com/yannikkellerde/TD3.git
RUN python3.7 -m pip install -r requirements.txt
RUN python3.7 -m pip install -e water-pouring/
RUN python3.7 -m pip install SPlisHSPlasH/

# Make partio work
RUN python3.7 -c "import os;os.makedirs('/root/.local/lib/python3.7/site-packages/',exist_ok=True)"
COPY docker_stuff/site-packages/ /root/.local/lib/python3.7/site-packages/
COPY docker_stuff/libpartio.so /usr/local/lib/
#RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/

#Run the program
WORKDIR /usr/src/bachelorthesis/TD3
CMD python3.7 main.py --env water_pouring:Pouring-mdp-full-v0