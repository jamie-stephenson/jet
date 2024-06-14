#!/bin/bash

#--RUN CONFIG SCRIPT--
config_file="./configs/cluster_config.sh"

if [ ! -f "$config_file" ]; then
    echo "Error: $config_file not found."
    exit 1
fi
source $config_file
#---------------------

#-----NTPUPDATE-------
sudo apt install ntpdate -y
#---------------------

#-----MOUNT DRIVE-----
sudo mkdir /clusterfs
sudo echo "$drive_addr /clusterfs  cifs  user=$drive_usr,password=$drive_pwd,rw,uid=1000,gid=1000,users 0 0" >> /etc/fstab
sudo mount /clusterfs
#---------------------

#-----CLONE REPO------

#---UPDATE HOSTNAME--- 
sudo hostname node00  
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

#-------SLURM---------
sudo apt install slurm-wlm -y
# Edit /etc/slurm/slurm.conf:
slurm_nodes=""
for node in "${nodes[@]}"; do
    name="${node}[name]"
    addr="${node}[addr]"
    cpus="${node}[cpus]"
    slurm_nodes+="NodeName=${!name} NodeAddr=${!addr} CPUs=${!cpus} State=UNKNOWN\n"
done

sudo sed -i "s/NodeName= NodeAddr= CPUs= State=UNKNOWN/$slurm_nodes/" /etc/slurm/slurm.conf
sudo sed -i "s/ClusterName= /ClusterName=$cluster_name/" /etc/slurm/slurm.conf

last_index=$(( ${#nodes[@]} - 1 )) # Note: this relies on strict naming pattern: node00 node01 node02 ...
node_string="node[01-$(printf "%02d" $last_index)]"

sudo sed -i "s/Nodes= /Nodes=$node_string/" /etc/slurm/slurm.conf
#---------------------
