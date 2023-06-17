#!/bin/bash

source /home/dell/anaconda3/etc/profile.d/conda.sh
conda activate py36
python train.py -type ACP -use_PSSM True -train_fasta ./data/ACPmain.txt -train_pssm ./data/pssm_acpmain/ -test_fasta ./data/ACPmaintest.txt -test_pssm ./data/pssm_acpmaintest/


