# Fossil-ID-Multiview-Deep-Ensembles
### Fossil Image Identification using Deep Learning Ensembles of Data Augmented Multiviews
Fossil identification is a substantial part of evolutionary studies. Automatic identification models, especially recent advances based on deep learning, heavily rely on the quantity and quality of labeled images to train the models. However, the images are particularly limited for paleontologists due to the fossil preservation, conditioned sampling, and expensive and inconsistent label annotation by domain experts. To address these challenges, we proposed an innovative multiview ensemble framework that collects the multiple views of each fossil specimen image of reflecting its different characteristics to train multiple base models and makes the final decision via soft voting. 

## Install
```bash
conda create -n MulEnsID python=3.10.8    
conda activate MulEnsID
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyyaml=6.0
conda install huggingface_hub=0.10.1
pip install opencv-python==4.6.0.66
pip install -U scikit-learn==1.1.1
conda install scikit-image=0.19.3
```

## Usage
#### ResNet-50
```bash
CUDA_VISIBLE_DEVICES=0 python train.py data/Original_img --lr 0.01 -b 32 --epochs 500 --output ./output/resnet50 --model resnet50 --num-classes 16 --pretrained  --experiment ori-view 
CUDA_VISIBLE_DEVICES=0 python train.py data/gray --lr 0.1 -b 64 --epochs 500 --output ./output/resnet50 --model resnet50 --num-classes 16 --pretrained  --experiment gray-view 
CUDA_VISIBLE_DEVICES=0 python train.py data/skeleton --lr 0.1 -b 32 --epochs 500 --output ./output/resnet50 --model resnet50 --num-classes 16 --pretrained  --experiment skeleton-view 
CUDA_VISIBLE_DEVICES=0 python voting.py --view1 data/Original_img/test/ --view2 data/gray/test/ --view3 data/skeleton/test/ -cp1 ./output/resnet50/ori-view/model_best.pth.tar -cp2 ./output/resnet50/gray-view/model_best.pth.tar -cp3 ./output/resnet50/skeleton-view/model_best.pth.tar --model resnet50  --num-classes 16
```
#### EfficientNet-b2
```bash
CUDA_VISIBLE_DEVICES=0 python train.py data/Original_img --lr 0.01 -b 32 --epochs 500 --output ./output/efficientnet-b2 --model efficientnet_b2 --num-classes 16 --pretrained  --experiment ori-view
CUDA_VISIBLE_DEVICES=0 python train.py data/gray --lr 0.1 -b 128 --epochs 500 --output ./output/efficientnet-b2 --model efficientnet_b2 --num-classes 16 --pretrained  --experiment gray-view 
CUDA_VISIBLE_DEVICES=0 python train.py data/skeleton --lr 0.1 -b 32 --epochs 500 --output ./output/efficientnet-b2 --model efficientnet_b2 --num-classes 16 --pretrained  --experiment skeleton-view
CUDA_VISIBLE_DEVICES=0 python voting.py --view1 data/Original_img/test/ --view2 data/gray/test/ --view3 data/skeleton/test/ -cp1 ./output/efficientnet-b2/ori-view/model_best.pth.tar -cp2 ./output/efficientnet-b2/gray-view/model_best.pth.tar -cp3 ./output/efficientnet-b2/skeleton-view/model_best.pth.tar --model efficientnet_b2  --num-classes 16
```

## Dataset
Please see the [README.md under the 2400_fus folder](https://github.com/houchengbin/Fossil-ID-Multiview-Deep-Ensembles/tree/main/2400_fus).

## Data Preprocessing
```bash
python datasplit.py --input ./2400_fus --output ./data/Original_img --train-rate 0.734 --val-rate 0.5  # seed=2022 with Microsoft Windows 11 version 21H2 for the data used in our paper
python multiview.py --input ./data/Original_img --output ./data --gray --binary --blocksize 41 --C 5 --skeletonize
```

## Reference
If you find [this work](https://arxiv.org/abs/2302.08062) is useful, please consider the following citation.
```
@article{hou2023fossil,
  title={Fossil Image Identification using Deep Learning Ensembles of Data Augmented Multiviews},
  author={Hou, Chengbin and Lin, Xinyu and Huang, Hanhui and Xu, Sheng and Fan, Junxuan and Shi, Yukun and Lv, Hairong},
  journal={arXiv preprint arXiv:2302.08062},
  year={2023}
}
```

