#!/bin/bash
drive_addr="$1"
drive_usr="$2"
drive_pwd="$3"
mount_dir="$4"

sudo apt-get update
echo  Filesystem type "none" given. Assuimg that filesystem is alread mounted at $mount_dir.