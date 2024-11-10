#!/bin/bash

# Tokenize: train tokenizer on dataset and then 
# use that tokenizer to encode that dataset.

# train tokenizer
JOBID1=$(sbatch scripts/sbatch/train_tokenizer.sh | awk '{print $4}')
echo "Train tokenizer job submitted with Job ID: $JOBID1"

# encode dataset
JOBID2=$(sbatch --dependency=afterok:${JOBID1} scripts/sbatch/encode.sh | awk '{print $4}')
echo "Encode dataset job submitted with Job ID: $JOBID2, dependent on Job ID: $JOBID1"
