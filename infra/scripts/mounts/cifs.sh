#!/bin/bash

drive_addr="$1"
drive_usr="$2"
drive_pwd="$3"
mount_dir="$4"

#-----MOUNT DRIVE-----
sudo apt-get update
sudo apt-get -o DPkg::Lock::Timeout=20 -y install cifs-utils
sudo mkdir $mount_dir
echo "$drive_addr $mount_dir cifs user=$drive_usr,password=$drive_pwd,rw,uid=1000,gid=1000,users 0 0" | sudo tee -a /etc/fstab >/dev/null
sudo mount $mount_dir
#---------------------