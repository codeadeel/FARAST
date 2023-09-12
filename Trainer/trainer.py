#!/usr/bin/env python3

# This file is used to train Face Detection Model & save it to ONNX format
# %%
# Importing Libraries
from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics_050_095
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
import torch
import os
import collections
collections.Iterable = collections.abc.Iterable

# %%
# Initializing Parameters
rootDir = os.environ.get('rootDir', '/data')
train_imgs_dir = f'{rootDir}/images/train'
train_labels_dir = f'{rootDir}/labels/train'
val_imgs_dir = f'{rootDir}/images/valid'
val_labels_dir = f'{rootDir}/labels/valid'
batchSize = int(os.environ.get('batchSize', 10))
epochs = int(os.environ.get('epochs', 10))
workers = int(os.environ.get('workers', 8))
classes = ['Face']

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': rootDir,
        'images_dir': train_imgs_dir,
        'labels_dir': train_labels_dir,
        'classes': classes
    },
    dataloader_params={'batch_size':batchSize, 'num_workers':workers}
)
 
val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': rootDir,
        'images_dir': val_imgs_dir,
        'labels_dir': val_labels_dir,
        'classes': classes
    },
    dataloader_params={'batch_size':batchSize, 'num_workers':workers}
)

train_params = {
    'silent_mode': False,
    "average_best_models":True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": epochs,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(classes),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(classes),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        ),
        DetectionMetrics_050_095(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(classes),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50:0.95'
}

# %%
# Initializing Training
 
CHECKPOINT_DIR = os.environ.get('saveDir', '/output')
 
trainer = Trainer(
    experiment_name='yolo_nas_s', 
    ckpt_root_dir=CHECKPOINT_DIR
)
 
model = models.get(
    'yolo_nas_s', 
    num_classes=len(classes), 
    pretrained_weights="coco"
)
 
trainer.train(
    model=model, 
    training_params=train_params, 
    train_loader=train_data, 
    valid_loader=val_data
)

# %%
# Model Save to ONNX
print('Model ONNX Conversion')
torch.cuda.empty_cache()
model = models.get('yolo_nas_s', num_classes=len(classes), checkpoint_path=f'{CHECKPOINT_DIR}/yolo_nas_s/ckpt_best.pth')
torch.onnx.export(model,
                  torch.randn(1, 3, 640, 640),
                  f'{CHECKPOINT_DIR}/nasSmall.onnx',
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
