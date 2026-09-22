import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('cfg/models/11/yolo11.yaml')
    model = YOLO('cfg/models/11/yolo11_WTConv.yaml')
    model.train(data='datasets/FloorPlanCAD/data.yaml',
                device=0,
                cache=False,
                imgsz=800,
                epochs=250,
                patience=80,
                batch=8,
                close_mosaic=30,
                optimizer='AdamW',
                lr0=0.001,
                lrf=0.01,
                weight_decay=0.0005,
                warmup_epochs=8,
                warmup_momentum=0.8,
                warmup_bias_lr=0.05,
                cos_lr=True,
                mosaic=0.3,
                mixup=0.0,
                copy_paste=0.0,
                hsv_h=0.0,
                hsv_s=0.1,
                hsv_v=0.2,
                degrees=2.0,
                translate=0.05,
                scale=0.25,
                shear=0.0,
                perspective=0.0,
                fliplr=0.5,
                flipud=0.0,
                amp=True,
                project='runs/train',
                name='exp_cad_hr_bifpn',
                )
