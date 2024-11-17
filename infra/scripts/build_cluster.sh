#!/bin/bash

#--CONFIGURE VARIABLES--
config_file=~/dtt/infra/configs/cluster_config.sh

if [ ! -f "$config_file" ]; then
    echo "Error: $config_file not found."
    exit 1
fi
source $config_file

slurm_conf_path=$mount_dir/dtt/infra/configs/slurm/
worker_script=$mount_dir/dtt/infra/scripts/build_worker.sh
env_script=$mount_dir/dtt/infra/scripts/env.sh
#---------------------

#-----MOUNT DRIVE----- TODO: support more fs types
mount_script=~/dtt/infra/scripts/mounts/$fs_type.sh 
mount_args="$drive_addr $drive_usr $drive_pwd $mount_dir"
source $mount_script $mount_args
#---------------------

#-----CLONE REPO------
sudo chown $USER $mount_dir
mkdir $mount_dir/dtt/
git clone https://github.com/jamie-stephenson/dtt.git $mount_dir/dtt/
#---------------------

#-EDIT SLURM CONFIGS--
slurm_nodes=""
gres_nodes=""
for node in "${nodes[@]}"; do
    name="${node}[name]"
    addr="${node}[addr]"
    cpus="${node}[cpus]"
    gpus="${node}[gpus]"
    slurm_nodes+="NodeName=${!name} NodeAddr=${!addr} CPUs=${!cpus} Gres=gpu:${!gpus} State=UNKNOWN\n"
    if ! [ ${!gpus} = 0 ]; then 
        gres_nodes+="NodeName=${!name} Name=gpu File=/dev/nvidia"
        gres_ids="0\n"
        if ! [ ${!gpus} = 1 ]; then 
            gres_ids="[0-$(($gpus - 1))]\n"
        fi
        gres_nodes+=$gres_ids
    fi        
done

sudo sed -i "s/NodeName= NodeAddr= CPUs= Gres= State=UNKNOWN/$slurm_nodes/" ${slurm_conf_path}slurm.conf
sudo sed -i "s/ClusterName=/ClusterName=$cluster_name/" ${slurm_conf_path}slurm.conf 

master_addr="${node00[addr]}"
sudo sed -i "s/SlurmctldHost=/SlurmctldHost=node00($master_addr)/" ${slurm_conf_path}slurm.conf

last_index=$(( ${#nodes[@]} - 1 )) # Note: this relies on strict naming pattern: node00 node01 node02 ...
node_string="node[01-$(printf "%02d" $last_index)] "

sudo sed -i "s/Nodes= /Nodes=$node_string/" ${slurm_conf_path}slurm.conf
sudo sed -i "s@NodeName= Name=gpu File=/dev/nvidia0@$gres_nodes@" ${slurm_conf_path}gres.conf
echo "$mount_dir*" >> ${slurm_conf_path}cgroup_allowed_devices_file.conf
#---------------------

#----SLURM LOGGING----
mkdir -p $mount_dir/dtt/slurmlogs/
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

#------NTPDATE--------
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 install ntpdate -y
#---------------------

#-------SLURM---------
sudo NEEDRESTART_MODE=l apt-get -o DPkg::Lock::Timeout=60 install slurm-wlm -y
sudo cp -r "${slurm_conf_path}"* /etc/slurm/
sudo cp /etc/munge/munge.key $mount_dir
sudo systemctl enable munge
sudo systemctl start munge
sudo systemctl enable slurmd
sudo systemctl start slurmd
sudo systemctl enable slurmctld
sudo systemctl start slurmctld
#---------------------

#----GNU PARALLEL-----
sudo apt-get -o DPkg::Lock::Timeout=60 -y install parallel
#---------------------

#--BUILD WORKERS--
touch ~/.ssh/known_hosts
chmod 600 ~/.ssh/known_hosts
mkdir ~/dtt_config_logs

run_on_node() {
    local node=$1
    local args=( $node "$hosts" $slurm_conf_path $mount_dir $env_script )

    output_file=~/dtt_config_logs/$node.log

    if ! [ $node = 'node00' ]; then
        if ! ssh-keygen -F $node; then
            ssh-keyscan -t ed25519 -H $node >> ~/.ssh/known_hosts
        fi
        ssh -i ~/.ssh/$key_name $USER@$node "bash -s -- $mount_args" < $mount_script > $output_file 2>&1
        ssh -i ~/.ssh/$key_name $USER@$node "bash -s -- ${args[@]@Q}" < $worker_script > $output_file 2>&1
    else        
        #--------ENV----------
        source $env_script $mount_dir
        #---------------------
    fi
}

# Export the function and vars to make them available to parallel
export -f run_on_node
export worker_script hosts slurm_conf_path mount_dir mount_script mount_args env_script key_name

# Run worker_script in parallel on all nodes
parallel -j 0 run_on_node {} ::: "${nodes[@]}"
#---------------------