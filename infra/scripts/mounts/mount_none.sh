#!/bin/bash
mount_dir="$1"
sudo apt-get update
echo  Filesystem type "none" given. Assuimg that filesystem is alread mounted at $mount_dir.