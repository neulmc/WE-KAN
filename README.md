# WE-KAN: A SAR Image Rotating Object Detection Method Based on Wavelet Domain Feature Enhancement and KAN Prediction Head

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12.1+](https://img.shields.io/badge/pytorch-1.12+-red.svg)](https://pytorch.org/)

This repository contains the implementation of our paper **"WE-KAN: A SAR Image Rotating Object Detection Method Based on Wavelet Domain Feature Enhancement and KAN Prediction Head"**, introducing a novel rotated object detection framework specifically designed for SAR images with complex backgrounds and strong noise.

## Key Features
- **Wavelet Domain Feature Enhancement**: Integrates wavelet scattering coefficients to enhance structural information and suppress speckle noise.
- **KAN-based Angle Predictor**: Utilizes Kolmogorov–Arnold Networks for more accurate and robust angle regression.
- **Shape-Aware Joint Loss**: Combines Rotated IoU Loss and Gaussian Distance Loss for precise bounding box regression.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12.1+
- CUDA 11+ (for GPU acceleration)

### Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/neulmc/WE-KAN.git
   cd WE-KAN
2. Install mmcv, mmdet, mmrotate.

   Please refer to [mmrotate dev-1.x](https://github.com/open-mmlab/mmrotate/blob/dev-1.x) for more detailed instructions,
   or [RSAR](https://github.com/zhasion/RSAR) official repository.
3. Install wavelet scattering toolkit.
   ```bash
   pip install kymatio
   
### Dataset Preparation
#### RSAR Dataset
We use the RSAR dataset, a large-scale rotated SAR object detection benchmark. It contains 95,842 images with 183,534 instances across six categories: Ship, Tank, Bridge, Aircraft, Harbor, and Car.

You can download the dataset from the official RSAR benchmark page or contact the authors.
#### Dataset Structure:
```
dataset/
    ├── RSAR/               # RSAR dataset
       ├── train/           # Training dataset (78,837 images)
           ├── images/      
           ├── annfiles/
       ├── val/             # Validation dataset (8,467 images)
           ├── images/      # 
           ├── annfiles/
       ├── test             # Test dataset (8,538 images)
           ├── images/
           ├── annfiles/

```

### Pipeline
#### 1. Train model

Key configurable parameters in train.py:
```
wekan_config = '..r50_fpn_1x_wavekan..'   # r50: ResNet50; r101: ResNet101
wekan_Atten = spatial_channel    # spatial_channel: WE-Attention; Cross: Cross-Attention
loss_weight = 0.5                # Loss weight in wekan_loss_RIoU; wekan_loss_GD
```
#### 2. Evaluation
After training, run vis_test.py:
```bash
python vis_test.py
```
#### 3. Visualization
The visualization runs automatically when evaluation, generating:
- Prediction visualizations in vis_results/filename.png
- Detailed metrics in detection_results/detection_results.json.


### Results on RSAR Dataset
| Method                   | Backbone    | Schedule | AP50  | AP75  | mAP  |
|--------------------------|-------------|----------|-------|-------|------|
| Rotated-RetinaNet        | ResNet50    | 1×       | 57.7  | 22.7  | 27.7 |
| R3Det                    | ResNet50    | 1×       | 63.9  | 25.0  | 30.5 |
| S2ANet                   | ResNet50    | 1×       | 66.5  | 28.5  | 33.1 |
| Rotated-Faster RCNN      | ResNet50 | 1×       | 63.2  | 24.9  | 30.5 |
| O-RCNN                   | ResNet50    | 1×       | 64.8  | 32.7  | 33.6 |
| ReDet                    | ReResNet50  | 1×       | 64.7  | 32.8  | 34.3 |
| RoI-Transformer          | ResNet50    | 1×       | 66.9  | 32.7  | 35.0 |
| Deformable DETR          | ResNet50    | 3×       | 46.6  | 13.1  | 19.6 |
| ARS-DETR                 | ResNet50    | 3×       | 61.1  | 29.0  | 31.6 |
| Rotated-FCOS             | ResNet50    | 1×       | 66.7  | 31.5  | 34.2 |
| Wave-KAN (**Ours**-R50)  | ResNet50    | 1×       | 70.1  | 32.5  | 35.9 |
| Wave-KAN (**Ours**-R101) | ResNet101   | 3×       | 74.6  | 32.8  | 37.5 |

### Final models
This is the pre-trained model and log file in our paper. We used this model for evaluation. You can download by:
https://pan.baidu.com/s/1t976AkkohviPFbYwlhPMOg?pwd=s63i code: s63i.

### Visualization
To better understand our method's performance, 
we provide visualization examples of rotated object detection results on SAR images. 
The visualization output can be found in the same model download link above.

### Code Structure
```
PSCL-github/
├── configs                 # Configuration files
│   ├── rotated_fcos         
├── mmrotate                # Execute code
│   ├── models/backbones    # Proposed WE-Attention      
│   ├── models/dense_heads       
│   └── models/layers       # Proposed KAN predictor 
├── RSAR                    # Dataset root
├── tests                   # Useful Evaluator 
├── tools              
│   ├── train.py            # Train our model
│   ├── train_R101.py       # Train our model (ResNet101)
│   ├── vis_test.py         # Test our model
│   └── vis_tool.py       
└── requirements            # Required packages
```

### References
[1] <a href="https://github.com/zhasion/RSAR">RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark.</a>

[2] <a href="https://github.com/open-mmlab/mmrotate">OpenMMLab Rotated Object Detection Toolbox and Benchmark.</a>

[3] <a href="https://github.com/Blealtan/efficient-kan">An Efficient Implementation of Kolmogorov-Arnold Network.</a>


