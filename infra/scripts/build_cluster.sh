#!/bin/bash

sudo apt-get update

#--RUN CONFIG SCRIPT--
config_file="cluster_config.sh"

if [ ! -f "$config_file" ]; then
    echo "Error: $config_file not found."
    exit 1
fi
source $config_file
#---------------------

#-----MOUNT DRIVE-----
sudo apt-get -o DPkg::Lock::Timeout=20 -y install cifs-utils
sudo mkdir /clusterfs
echo "$drive_addr /clusterfs cifs user=$drive_usr,password=$drive_pwd,rw,uid=1000,gid=1000,users 0 0" | sudo tee -a /etc/fstab >/dev/null
sudo mount /clusterfs
#---------------------

#-----CLONE REPO------
sudo chown $USER /clusterfs
mkdir /clusterfs/jet/
git clone https://github.com/jamie-stephenson/jet.git /clusterfs/jet/
#---------------------

#---EDIT SLURM.CONF---
slurm_nodes=""
for node in "${nodes[@]}"; do
    name="${node}[name]"
    addr="${node}[addr]"
    cpus="${node}[cpus]"
    slurm_nodes+="NodeName=${!name} NodeAddr=${!addr} CPUs=${!cpus} State=UNKNOWN\n"
done

sudo sed -i "s/NodeName= NodeAddr= CPUs= State=UNKNOWN/$slurm_nodes/" ${slurm_conf_path}slurm.conf
sudo sed -i "s/ClusterName=/ClusterName=$cluster_name/" ${slurm_conf_path}slurm.conf 

master_addr="${node00[addr]}"
sudo sed -i "s/SlurmctldHost=/SlurmctldHost=node00($master_addr)/" ${slurm_conf_path}slurm.conf

last_index=$(( ${#nodes[@]} - 1 )) # Note: this relies on strict naming pattern: node00 node01 node02 ...
node_string="node[01-$(printf "%02d" $last_index)] "

sudo sed -i "s/Nodes= /Nodes=$node_string/" ${slurm_conf_path}slurm.conf
#---------------------

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
sudo sed -i 's/.*/node00/' /etc/hostname
sudo sed -i 's/^preserve_hostname: false$/preserve_hostname: true/' /etc/cloud/cloud.cfg
#---------------------

#-----NTPUPDATE-------
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=20 install ntpdate -y
#---------------------

#-------SLURM---------
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 install slurm-wlm -y
sudo cp "${slurm_conf_path}slurm.conf" "${slurm_conf_path}cgroup.conf" "${slurm_conf_path}cgroup_allowed_devices_file.conf" /etc/slurm/
sudo cp /etc/munge/munge.key /clusterfs
sudo systemctl enable munge
sudo systemctl start munge
sudo systemctl enable slurmd
sudo systemctl start slurmd
sudo systemctl enable slurmctld
sudo systemctl start slurmctld
#---------------------

#----GNU PARALLEL-----
sudo apt-get -o DPkg::Lock::Timeout=20 -y install parallel
#---------------------

#--RUN WORKER CONFIG--
touch ~/.ssh/known_hosts
chmod 600 ~/.ssh/known_hosts
mkdir ~/jet_config_logs

run_on_node() {
    local node=$1
    local script=$2
    shift 2
    local args=( "$node" "$@" )

    output_file=~/jet_config_logs/$node.log

    if ! [ $node = 'node00' ]; then
        if ! ssh-keygen -F $node; then
            ssh-keyscan -t ed25519 -H $node >> ~/.ssh/known_hosts
        fi
        ssh -i ~/.ssh/id_ed25519 $USER@$node "bash -s -- ${args[@]@Q}" < $script > $output_file 2>&1
    fi
}

# Export the function to make it available to parallel
export -f run_on_node

args=( "$worker_script" "$drive_addr" "$drive_usr" "$drive_pwd" "$hosts" "$slurm_conf_path" "$torch_index" )
# Run worker_script in parallel on all nodes
parallel -j 0 run_on_node {} "${args[@]@Q}" ::: "${nodes[@]}"
#---------------------

#-PYTHON ENVIRONMENT--
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get -o DPkg::Lock::Timeout=20 -y install python3.11
sudo apt-get -o DPkg::Lock::Timeout=60 -y install python3.11-venv
mkdir envs
python3.11 -m venv ~/envs/jet
source ~/envs/jet/bin/activate
pip install -r /clusterfs/jet/requirements.txt
pip install torch --index-url $torch_index
deactivate
mkdir -p /clusterfs/jet/logs/
#---------------------