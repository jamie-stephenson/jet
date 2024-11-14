#!/bin/bash
# Script to set up local enivronment on each node 

mount_dir="$1"

#-PYTHON ENVIRONMENT--
sudo NEEDRESTART_MODE=l add-apt-repository -y ppa:deadsnakes/ppa
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 -y install python3.11
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 -y install python3.11-venv
python3.11 -m venv ~/envs/jet
source ~/envs/jet/bin/activate
pip install -r $mount_dir/jet/requirements.txt
pip install torch --index-url $torch_index
deactivate
#---------------------     

#------OPEN MPI-------
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 install -y openmpi-bin openmpi-common libopenmpi-dev 
#---------------------