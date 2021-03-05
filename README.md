# Water-pouring Gym environment

This repository contains a water-pouring Reinforcement Learning environment implemented in OpenAI gym.

## Installation
Installation has been tested an Ubuntu 18.04 machine. Other Debian based distributions shoud work as well, otherwise you are probably out of luck. Because this project uses the fluid-simulator SPlisHSPlasH there are a few packages that are required to get going. There is a [dockerfile](dockerfile) provided that shows all steps required for a successfull installation. If that does not work for you, the following are the dependencies:

1. python3.7, other versions might work as well
2. The following apt packages: cmake, build-essential, libx11-dev, xorg-dev, libglu1-mesa-dev, python3.7-dev, python3-setuptools
3. The python packages in requirements.txt. I would recommend using a virtualenv.
4. My SPlisHSPlasH fork on https://github.com/yannikkellerde/SPlisHSPlasH. Install by cloning the repo and running `pip install SPlisHSPlasH/`. Installation might take a few minutes.
5. Partio's python bindings https://github.com/wdas/partio. Installation of the python bindings can be a little tricky. I would recommend just copying the libpartio.so libary that is provided in the [docker_stuff](docker_stuff/) folder in this repo to a location included in your LD_LIBRARY_PATH and copying the files from [docker_stuff/site_packages](docker_stuff/site_packages) to your virtualenv site-packages.

Now you can install the water-pouring environment with `pip install -e water-pouring/` from this repositories root.

## Testing the installation
Run `python human_player` to test the installation. In this test program, you can control the bottle yourself using the arrow keys and W/S.

## Provided environments
**water_pouring:Pouring-mdp-v0** - Pour from a bottle into a glass while getting the full simulators state as observations.
**water_pouring:Pouring-featured-v0** - Pour from a bottle into a glass while getting handcrafted features as observations that do not fully describe the state of the simulator.
**water_pouring:Pouring-g2g-mdp-v0** - Pour from a glass into another glass while getting the full simulators state as observations.
**water_pouring:Pouring-featured-v0** - Pour from a glass into another glass while getting handcrafted features as observations.