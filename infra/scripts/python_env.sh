#!/bin/bash

torch_index="$1"
mount_dir="$2"

#-PYTHON ENVIRONMENT--
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get -o DPkg::Lock::Timeout=60 -y install python3.11
sudo apt-get -o DPkg::Lock::Timeout=60 -y install python3.11-venv
python3.11 -m venv ~/envs/jet
source ~/envs/jet/bin/activate
pip install -r $mount_dir/jet/requirements.txt
pip install torch --index-url $torch_index
deactivate
#---------------------