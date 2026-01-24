# SparseCoop: Cooperative Perception with Kinematic-Grounded Queries

## Quick Start

### Set up a new virtual environment
```bash
##### conda or mamba is recommended
conda create -n sparsecoop python=3.8
conda activate sparsecoop

##### Or use virtualenv
# virtualenv mm_sparsecoop --python=python3.8
# source mm_sparsecoop/bin/activate
```

### Install packpages using pip3
```bash

#### Install pytorch following: https://pytorch.org/get-started/previous-versions/
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

#### Install mmcv series following: https://mmcv-zh-cn.readthedocs.io/zh-cn/v1.7.1/get_started/installation.html
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
pip install mmdet==2.28.2

#### Install other packages
pip3 install -r requirement.txt
```

### Compile the deformable_aggregation CUDA op
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
```

### Prepare data (V2X-Seq-SPD)

Downlowd [V2X-Seq-SPD](https://thudair.baai.ac.cn/coop-forecast) dataset, organize the files as official instructions and then follow the steps below to prepare the dataset.

a. Create V2X-Seq-SPD-Example (as an example, you can give it a name that you like) according to given sequence ids. Meanwhile clean the dataset to remove frames which don't have valid ground truth and optimize label in several frames.

```bash
python ./tools/spd_data_converter/gen_example_data.py --input YOUR_V2X-Seq-SPD_ROOT --output ./datasets/V2X-Seq-SPD-2hz-for-SparseCoop --sequences 'all' --update-label --freq 2
```

b. Generate data info for training and nuscenes style dataset for evaluation. 

```bash
bash ./tools/spd_example_converter.sh V2X-Seq-SPD-2hz-for-SparseCoop vehicle-side
bash ./tools/spd_example_converter.sh V2X-Seq-SPD-2hz-for-SparseCoop infrastructure-side
bash ./tools/spd_example_converter.sh V2X-Seq-SPD-2hz-for-SparseCoop cooperative
```

### Prepare data (Griffin)
Download the [Griffin dataset](https://github.com/wang-jh18-SVM/Griffin) and create symbolic links.
```bash
mkdir datasets
ln -s path/to/griffin_50scenes_25m ./datasets/griffin_50scenes_25m
```

Pack the meta-information and labels of the dataset, and generate the required .pkl files and anchors.
```bash
bash tools/griffin_converter.sh griffin_50scenes_25m
```

### Download pre-trained weights
Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).
```bash
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

### Training and testing
```bash
# train single side models
bash local_train.sh projects/configs/V2X-Seq-SPD-2hz/vehicle-side/sparsecoop_v1-1_r50_544x960_bs6x8_800e.py
bash local_train.sh projects/configs/V2X-Seq-SPD-2hz/infrastructure-side/sparsecoop_v1-1_r50_544x960_bs6x8_800e.py

# test infrastructure side model and save queries
bash local_test.sh projects/configs/V2X-Seq-SPD-2hz/infrastructure-side/sparsecoop_v1-1_r50_544x960_bs6x8_800e_no_aug_eval_train_dn.py path/to/checkpoint
bash local_test.sh projects/configs/V2X-Seq-SPD-2hz/infrastructure-side/sparsecoop_v1-1_r50_544x960_bs6x8_800e_no_aug_eval.py path/to/checkpoint

# train cooperative model (remember to update the load_from param in config file)
bash local_train.sh projects/configs/V2X-Seq-SPD-2hz/cooperative/sparsecoop_v1-4_r50_544x960_bs4x8_48e_no_aug_co153CT.py

# other config files are for ablation experiments, for example
bash local_train.sh projects/configs/V2X-Seq-SPD-2hz/cooperative/sparsecoop_v1-4_r50_544x960_bs4x8_48e_no_aug_co150_woRefine.py
```

## Acknowledgement
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [DETR3D](https://github.com/WangYueFt/detr3d)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [Sparse4D](https://github.com/HorizonRobotics/Sparse4D)