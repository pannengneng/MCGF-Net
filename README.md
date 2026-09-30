# PyTorch Code for MCGF-Net

**MCGF-Net: Multi-Dimensional Collaborative Geometric Feature Enhancement Network for Object Detection in CAD Drawings**

MCGF-Net is an object detection network designed for component detection in CAD drawings. 
It enhances geometric feature representation through multi-domain, spatial-channel, and geometry-aware feature enhancement, improving the detection of small-scale, densely distributed, and easily confused components in complex CAD drawings.

## Requirement

This code is implemented based on PyTorch and Ultralytics YOLO11.

The main requirements include:

- Python >= 3.8
- PyTorch
- torchvision
- ultralytics
- CUDA (recommended for GPU training)

Install the required packages:

```bash
pip install -r requirements.txt


1. Dataset
The experiments are conducted on the FloorPlanCAD dataset.
Please organize the dataset in YOLO format and place it under datasets.
The recommended directory structure is:
datasets/
└── FloorPlanCAD/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/

Configure the dataset path and category information in the corresponding YAML file before training.


2. Train
To train MCGF-Net, run:
python train.py

The training parameters, dataset path, model configuration, batch size, image size, and number of epochs can be configured in train.py or the corresponding configuration files.


3. Test
After training, evaluate the trained model using the generated model weights.
For example:
python main.py

Please modify the model weight path and dataset configuration according to your local environment.


Reference
If you find this project useful for your research, please cite our work:
@inproceedings{MCGFNet2026,
  title={MCGF-Net: Multi-Dimensional Collaborative Geometric Feature Enhancement Network for Object Detection in CAD Drawings},
  author={Pan, Yuting and Jiang, Zhongmin and Ouyang, Xiong and Guo, Ziyang and Wang, Wenju},
  year={2026}
}


Acknowledgement
This project is developed based on the Ultralytics YOLO framework.
We sincerely thank the authors and contributors of the related open-source projects for their excellent work.
