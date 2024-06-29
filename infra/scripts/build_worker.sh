#!/bin/bash

node="$1"
hosts="$2"
slurm_conf_path="$3"
torch_index="$4"
mount_dir="$5"
python_env_script="$6"

#---UPDATE HOSTNAME--- 
sudo hostname $node  
sudo sed -i 2d /etc/hosts
sudo sed -i "2i $hosts" /etc/hosts 
sudo sed -i "s/.*/$node/" /etc/hostname
sudo sed -i 's/^preserve_hostname: false$/preserve_hostname: true/' /etc/cloud/cloud.cfg
#---------------------

#-----NTPUPDATE-------
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=20 install ntpdate -y
#---------------------

#-------SLURM---------
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 install slurmd slurm-client -y
sudo cp $mount_dir/munge.key /etc/munge/munge.key
sudo cp -r "${slurm_conf_path}"* /etc/slurm/
sudo systemctl enable munge
sudo systemctl start munge
sudo systemctl enable slurmd
sudo systemctl start slurmd
#---------------------

#-----PYTHON ENV------
source $python_env_script $torch_index $mount_dir
#---------------------