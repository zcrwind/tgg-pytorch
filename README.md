# :fire: TGG

The source code of our ACM MM 2019 conference paper ["TGG: Transferable Graph Generation for Zero-shot and Few-shot Learning"](/docs/paper/acmmm2019_tgg.pdf).

![TGG framework](/docs/imgs/framework.png)


## Requirements
- python 3.7.1+
- pytorch 1.0.0+
- numpy 1.15.4+
- scipy 1.2.0+
- scikit-learn 0.21.2+
- requests 2.21.0+
- matplotlib 3.0.2+
- CUDA 10.0+
- cudnn 6.0.21+


## Datasets
- **aPY**. Attribute Pascal and Yahoo (aPY) is a small-scale coarse-grained dataset with 64 attributes.
- **CUB** Caltech-UCSDBirds 200-2011 (CUB) is a fine-grained and medium scale dataset with respect to both number of images and number of classes, i.e. 11, 788 images from 200 different types of birds annotated with 312 attributes.
- **AwA1** Animals with Attributes (AWA1) is a coarse-grained dataset that is medium-scale in terms of the number of images, i.e. 30, 475 and small-scale in terms of number of classes, i.e. 50 classes.
- **AwA2** Animals with Attributes2 (AWA2) is introduced by [9], which contains 37, 322 images for the 50 classes of AWA1 dataset from public web sources, i.e. Flickr, Wikipedia, etc., making sure that all images of AWA2 have free-use and redistribution licenses and they do not overlap with images of the original Animal with Attributes dataset.
- **SUN** SUN is a fine-grained and medium-scale dataset with respect to both number of images and number of classes, i.e. SUN contains 14340 images coming from 717 types of scenes annotated with 102 attributes.

**NOTE**: our TGG algorithm is feature-agnostic, hence you can use any type of visual feature as input (In our implement, following [1], ResNet101 feature is used for a fair comparison).

## Usage
Download the datasets from [here](http://www.robots.ox.ac.uk/~lz/DEM_cvpr2017/data.zip) and put them into the `tgg-pytorch/data/`, then
### 1. Build the class-level graphs with [ConceptNet5.5](http://www.conceptnet.io/)
```
$ cd tgg-pytorch/preprocess/
$ python graph_construction.py
```
The class-level graph (pickle file) will be saved at `tgg-pytorch/data/preprocessed_data/${dataset_name}`. In this repo, we use two small datasets (i.e., aPY and AwA) for two fast examples, and the preprocessed class-level files are available at:
```
tgg-pytorch/data/preprocessed_data/apy/apy_class_adj_byConceptNet5.5_.pkl
tgg-pytorch/data/preprocessed_data/awa/awa_class_adj_byConceptNet5.5_.pkl
```
and their visualization is shown below:
![class-level graphs](/docs/imgs/class_level_graphs.png)

### 2. Train GAN models with the code in `tgg-pytorch/AUFS_ZSL`
Here `AUFS_ZSL` is our another work that is published in IJCAI 2018. See [https://github.com/zcrwind/AUFS_ZSL](https://github.com/zcrwind/AUFS_ZSL) for more details. For convenience, similarly, we provide two pre-trained AUFS models at:
```
tgg-pytorch/gan_checkpoints/apy/checkpoint_apy_iter951_accUnseen29.80_accSeen64.53.pkl
tgg-pytorch/gan_checkpoints/awa/checkpoint_awa_iter8401accUnseen57.34_accSeen72.49.pkl
```

### 3. Train and evaluate our TGG model:

for **aPY** dataset example:
```
$ sh main_aPY.sh
```
for **AwA** dataset example:
```
$ sh main_AwA1.sh
```

**NOTE**: modify hyperparameters in config files as needed, where the suitable learning rate is of great importance.

## Experimental Results
Due to space constraints, we refer the readers to our [paper](/docs/paper/acmmm2019_tgg.pdf) for the results of conventional ZSL, GZSL and FSL.

## References
[1] Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M Hospedales. 2018. Learning to compare: Relation network for few-shot learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 1199–1208.


If you make use of this code in your work, please cite the paper:
```
@inproceedings{zhang2019tgg,
        title={TGG: Transferable Graph Generation for Zero-shot and Few-shot Learning},
        author={Zhang, Chenrui and Lyu, Xiaoqing and Tang, Zhi},
        booktitle={ACM Conference on Multimedia},
        pages={1641--1649},
        year={2019}
}
```
