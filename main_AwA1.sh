#!/bin/bash

### dataset specific config >>>>>>>>>
dataset_name='awa'
resume='pass'
weight_threshold=0.2
gan_checkpoint='checkpoint_awa_iter8401accUnseen57.34_accSeen72.49.pkl'
### dataset specific config <<<<<<<<<


### shared config >>>>>>>>>
mode='train'
gpuid=1
n_iteration=7000    # 7000 iterations are enough for AwA1 dataset.
batch_size=512
lr=0.0005    # 0.0001、 0.001、 0.01 are bad, 0.0005 is the right initial learning rate!

weight_decay=1e-2
optimizer='adam'

labelIdxStart0or1=1
root_dir='./data'
graph_datadir='./data/preprocessed_data'
save_dir='./checkpoints'
all_visualFea_label_file='res101.mat'
auxiliary_file='original_att_splits.mat'

use_z='true'
z_dim=100

n_generation_perClass=50
classifier_type='softmax'
n_epoch_sftcls=100
use_pca='false'
reduced_dim_pca=1024

gan_checkpoint_dir='./gan_checkpoints'

print_every=10
eval_every=100
n_gene_perC=40
### shared config <<<<<<<<<


python main.py \
	--mode ${mode} \
	--dataset_name ${dataset_name} \
	--resume ${resume} \
	--n_iteration ${n_iteration} \
	--batch_size ${batch_size} \
	--lr ${lr} \
	--weight_decay ${weight_decay} \
	--optimizer ${optimizer} \
	--labelIdxStart0or1 ${labelIdxStart0or1} \
	--root_dir ${root_dir} \
	--graph_datadir ${graph_datadir} \
	--save_dir ${save_dir} \
	--all_visualFea_label_file ${all_visualFea_label_file} \
	--auxiliary_file ${auxiliary_file} \
	--gpuid ${gpuid} \
	--n_generation_perClass ${n_generation_perClass} \
	--classifier_type ${classifier_type} \
	--n_epoch_sftcls ${n_epoch_sftcls} \
	--use_pca ${use_pca} \
	--reduced_dim_pca ${reduced_dim_pca} \
	--weight_threshold ${weight_threshold} \
	--gan_checkpoint_dir ${gan_checkpoint_dir} \
	--gan_checkpoint ${gan_checkpoint} \
	--use_z ${use_z} \
	--z_dim ${z_dim} \
	--print_every ${print_every} \
	--eval_every ${eval_every} \
	--n_gene_perC ${n_gene_perC} \
