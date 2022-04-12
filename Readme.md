# setup

```
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python -m pip install git+https://github.com/cocodataset/panopticapi.git


wget https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/Panoptic-DeepLab/train_net.py
wget https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/Panoptic-DeepLab/configs/Cityscapes-PanopticSegmentation/Base-PanopticDeepLab-OS16.yaml
wget https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/Panoptic-DeepLab/configs/COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml
wget https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv/model_final_5e6da2.pkl
 
 
sed --in-place -e "s/..\/Cityscapes-PanopticSegmentation\///" panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml
```

# run

python main_gui.py output/


# usage

drag & drop an image to the window.
