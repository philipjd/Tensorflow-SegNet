#!/bin/bash

# for training
python main.py --log_dir=log --image_dir=../SegNet-Tutorial/CamVid/train.txt --val_dir=../SegNet-Tutorial/CamVid/val.txt --batch_size=5

# for finetune from saved ckpt
# python main.py --finetune=log/model.ckpt-19999  --log_dir=log --image_dir=../SegNet-Tutorial/CamVid/train.txt --val_dir=../SegNet-Tutorial/CamVid/val.txt --batch_size=5

# for testing
# python main.py --testing=log/model.ckpt-19999  --log_dir=log --test_dir=../SegNet-Tutorial/CamVid/test.txt --batch_size=5 --save_image=True

# infer images
# python main.py --infer=log/model.ckpt-19999  --log_dir=log --test_dir=/home/hao/data/kitti/raw/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/files.txt --save_image=True
