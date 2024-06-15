#!/bin/bash

node=$1

sudo apt-get update

#-----MOUNT DRIVE-----
sudo apt-get -y install cifs-utils
sudo mkdir /clusterfs
echo "$drive_addr /clusterfs cifs user=$drive_usr,password=$drive_pwd,rw,uid=1000,gid=1000,users 0 0" | sudo tee -a /etc/fstab
sudo mount /clusterfs
#---------------------

#--RUN CONFIG SCRIPT--
config_file="/clusterfs/jet/configs/cluster_config.sh"

if [ ! -f "$config_file" ]; then
    echo "Error: $config_file not found."
    exit 1
fi
source $config_file
#---------------------

#---UPDATE HOSTNAME--- 
sudo hostname $node  
sudo sed -i 2d /etc/hosts
hosts=""
for node in "${nodes[@]}"; do
    name="${node}[name]"
    addr="${node}[addr]"
    hosts+="${!addr} ${!name}\n"
done
sudo sed -i "2i $hosts" /etc/hosts 
sudo sed -i 's/.*/node00/' ./etc/hostname
sudo sed -i 's/^preserve_hostname: false$/preserve_hostname: true/' /etc/cloud/cloud.cfg
#---------------------

#-----NTPUPDATE-------
sudo NEEDRESTART_MODE=a apt-get install ntpdate -y
#---------------------

#-------SLURM---------
sudo NEEDRESTART_MODE=a apt-get install slurmd slurm-client -y
sudo cp /clusterfs/munge.key /etc/munge/munge.key
sudo cp "${slurm_conf_path}slurm.conf" /etc/slurm/slurm.conf
sudo cp "${slurm_conf_path}cgroup*" /etc/slurm
sudo systemctl enable munge
sudo systemctl start munge
sudo systemctl enable slurmd
sudo systemctl start slurmd
#---------------------

#-PYTHON ENVIRONMENT--
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get -y install python3.11
sudo apt-get -y install python3.11-venv
mkdir envs
python3.11 -m venv ~/envs/jet
source ~/envs/jet/bin/activate
pip install -r /clusterfs/jet/requirements.txt
pip install torch --index-url $torch_index
#---------------------