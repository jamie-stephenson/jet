#!/bin/bash

drive_addr="$1"
drive_usr="$2"
drive_pwd="$3"
mount_dir="$4"

#-----MOUNT DRIVE-----
sudo apt-get update
sudo apt-get -o DPkg::Lock::Timeout=20 -y install nfs-common 
sudo mkdir -p $mount_dir
echo "$drive_addr:/ $mount_dir nfs4 nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport,_netdev 0 0" | sudo tee -a /etc/fstab >/dev/null
sudo mount $mount_dir
#---------------------

# Verify the mount was successful
if mount | grep $mount_dir > /dev/null; then
    echo "NFS successfully mounted to $mount_dir"
else
    echo "Failed to mount NFS $drive_addr"
    exit 1
fi


