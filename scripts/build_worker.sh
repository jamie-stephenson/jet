#!/bin/bash

node="$1"
drive_addr="$2"
drive_usr="$3"
drive_pwd="$4"
hosts="$5"
slurm_conf_path="$6"
torch_index="$7"

wait_for_pkg_mngmt() {
    echo "Checking for running package management processes..."
    while sudo lsof /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/cache/apt/archives/lock >/dev/null 2>&1; do
        echo "Package management processes are running. Waiting for them to finish..."
        sleep 5
    done
    echo "No package management processes running. Continuing with the script..."
}

sudo apt-get update
wait_for_pkg_mngmt

#-----MOUNT DRIVE-----
sudo apt-get -y install cifs-utils
sudo mkdir /clusterfs
echo "$drive_addr /clusterfs cifs user=$drive_usr,password=$drive_pwd,rw,uid=1000,gid=1000,users 0 0" | sudo tee -a /etc/fstab >/dev/null
sudo mount /clusterfs
#---------------------

#---UPDATE HOSTNAME--- 
sudo hostname $node  
sudo sed -i 2d /etc/hosts
sudo sed -i "2i $hosts" /etc/hosts 
sudo sed -i "s/.*/$node/" /etc/hostname
sudo sed -i 's/^preserve_hostname: false$/preserve_hostname: true/' /etc/cloud/cloud.cfg
#---------------------

#-----NTPUPDATE-------
sudo NEEDRESTART_MODE=l apt-get install ntpdate -y
#---------------------

#-------SLURM---------
sudo NEEDRESTART_MODE=l apt-get install slurmd slurm-client -y
sudo cp /clusterfs/munge.key /etc/munge/munge.key
sudo cp -r "${slurm_conf_path}"* /etc/slurm/
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
deactivate
#---------------------