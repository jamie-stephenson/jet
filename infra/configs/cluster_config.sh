#!/bin/bash

#---Name---
cluster_name='jumbo'

#---Key---
key_name='name'

#---Filesystem---
fs_type=cifs
mount_dir=/clusterfs
drive_addr='address'
drive_usr='user'
drive_pwd='pwd'

#---Python-Env---
torch_index=https://download.pytorch.org/whl/cpu

#---Nodes---
nodes=()

declare -A node00=(
    [name]="node00"
    [addr]=12:34:56:0
    [cpus]=1
)
nodes+=(node00)

declare -A node01=(
    [name]="node01"
    [addr]=12:34:56:1
    [cpus]=8
)
nodes+=(node01)

declare -A node02=(
    [name]="node02"
    [addr]=12:34:56:2
    [cpus]=8
)
nodes+=(node02)

declare -A node03=(
    [name]="node03"
    [addr]=12:34:56:3
    [cpus]=8
)
nodes+=(node03)