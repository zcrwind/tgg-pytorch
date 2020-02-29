#!/bin/bash

mode='train'
gpuid=0
dataset_name='apy'
resume='pass'

n_iteration=10000
batch_size=64
lr_G=1e-4
lr_D=1e-4
lr_R=1e-4
weight_decay=1e-2
optimizer='adam'

labelIdxStart0or1=1
root_dir='../data'
save_dir='../gan_checkpoints'
all_visualFea_label_file='res101.mat'
auxiliary_file='original_att_splits.mat'
use_z='true'
z_dim=100


centroid_lambda=1
_lambda=0.00015
gp_lambda=10
regression_lambda=1

n_iter_D=1
n_iter_G=5

n_generation_perClass=50
classifier_type='softmax'
n_epoch_sftcls=100
use_pca='false'
reduced_dim_pca=1024

python main.py \
	--mode ${mode} \
	--dataset_name ${dataset_name} \
	--resume ${resume} \
	--n_iteration ${n_iteration} \
	--batch_size ${batch_size} \
	--lr_G ${lr_G} \
	--lr_D ${lr_D} \
	--lr_D ${lr_R} \
	--weight_decay ${weight_decay} \
	--optimizer ${optimizer} \
	--labelIdxStart0or1 ${labelIdxStart0or1} \
	--root_dir ${root_dir} \
	--save_dir ${save_dir} \
	--all_visualFea_label_file ${all_visualFea_label_file} \
	--auxiliary_file ${auxiliary_file} \
	--use_z ${use_z} \
	--z_dim ${z_dim} \
	--gpuid ${gpuid} \
	--centroid_lambda ${centroid_lambda} \
	--_lambda ${_lambda} \
	--gp_lambda ${gp_lambda} \
	--regression_lambda ${regression_lambda} \
	--n_iter_D ${n_iter_D} \
	--n_iter_G ${n_iter_G} \
	--n_generation_perClass ${n_generation_perClass} \
	--classifier_type ${classifier_type} \
	--n_epoch_sftcls ${n_epoch_sftcls} \
	--use_pca ${use_pca} \
	--reduced_dim_pca ${reduced_dim_pca}
