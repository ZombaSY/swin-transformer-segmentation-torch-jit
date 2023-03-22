# Swin-transformer conversion for torch.jit.save in segmentation task
This repository is for soft conversion of Swin-transformer for torch.jit.save in segmentation task.

Run "main.py" first to check that your model is convertible for torch.jit 

To load pretrained model from official weight, use model.load_pretrained_imagenet() to remove additional layer for segmentation.

You can download pretrained Swin-T on ImageNet-1K
[Here](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)

## Requirements
To install requirements:
```
pip install -r requirements.txt
```

