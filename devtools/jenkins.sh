#!/bin/bash
bash devtools/install.sh

export PATH=`pwd`/anaconda/bin:$PATH
source activate pml_qc

# Untar the checkpoints
cd checkpoint_files
tar xzf checkpoint_PML_QC_CCSD.tar.gz
cd ..

# Grab condition files from https://searchworks.stanford.edu/view/kf921gd3855
# https://stacks.stanford.edu/file/druid:kf921gd3855/minibatches_split_based_on_3rd_digit_9ha_batchsize16_batch95X.tar.gz
rsync -avz boltio:/nfs/working/deep_learn/key_store/pml_qc/minibatches_split_based_on_3rd_digit_9ha_batchsize16 ccsdt_dataset/

python scripts/pml_qc.py --mode predict \
                --checkpoint_dir checkpoint_files/checkpoint_PML_QC_CCSD \
                --condition_files "ccsdt_dataset/minibatches_split_based_on_3rd_digit_9ha_batchsize16/testing2_conditions_batch95*.npy"
