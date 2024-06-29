#!/bin/bash

#--CONFIGURE VARIABLES--
config_file=~/jet/infra/configs/cluster_config.sh

if [ ! -f "$config_file" ]; then
    echo "Error: $config_file not found."
    exit 1
fi
source $config_file

slurm_conf_path=$mount_dir/jet/infra/configs/slurm/
worker_script=$mount_dir/jet/infra/scripts/build_worker.sh
python_env_script=$mount_dir/jet/infra/scripts/python_env.sh
#---------------------

#-----MOUNT DRIVE----- TODO: support more fs types
mount_script=~/jet/infra/scripts/mount_$fs_type.sh 
if [ $fs_type = 'cifs' ]; then
    mount_args="$drive_addr $drive_usr $drive_pwd $mount_dir"
    source $mount_script $mount_args
fi
#---------------------

#-----CLONE REPO------
sudo chown $USER $mount_dir
mkdir $mount_dir/jet/
git clone -b slurm-tests https://github.com/jamie-stephenson/jet.git $mount_dir/jet/
#---------------------

#-EDIT SLURM CONFIGS--
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
echo "$mount_dir*" >> ${slurm_conf_path}cgroup_allowed_devices_file.conf
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
sudo cp /etc/munge/munge.key $mount_dir
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

#--BUILD WORKERS--
touch ~/.ssh/known_hosts
chmod 600 ~/.ssh/known_hosts
mkdir ~/jet_config_logs

run_on_node() {
    local node=$1
    local args=( $node "$hosts" $slurm_conf_path $torch_index $mount_dir $python_env_script )

    output_file=~/jet_config_logs/$node.log

    if ! [ $node = 'node00' ]; then
        if ! ssh-keygen -F $node; then
            ssh-keyscan -t ed25519 -H $node >> ~/.ssh/known_hosts
        fi
        ssh -i ~/.ssh/id_ed25519 $USER@$node "bash -s -- $mount_args" < $mount_script > $output_file 2>&1
        ssh -i ~/.ssh/id_ed25519 $USER@$node "bash -s -- ${args[@]@Q}" < $worker_script > $output_file 2>&1
    else
        source $python_env_script $torch_index $mount_dir
        mkdir -p $mount_dir/jet/logs/
    fi
}

# Export the function to make it available to parallel
export -f run_on_node


# Run worker_script in parallel on all nodes
parallel -j 0 run_on_node {} ::: "${nodes[@]}"
#---------------------