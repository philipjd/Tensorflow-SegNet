#!/bin/bash

# for training
python main.py --log_dir=log --image_dir=/home/dijiang/Work/data/public/kitti_road/data_road/training/train.txt --val_dir=/home/dijiang/Work/data/public/kitti_road/data_road/training/val.txt --batch_size=10

# for finetune from saved ckpt
# python main.py --finetune=log/model.ckpt-19999  --log_dir=log --image_dir=../SegNet-Tutorial/CamVid/train.txt --val_dir=../SegNet-Tutorial/CamVid/val.txt --batch_size=5

# for testing
# python main.py --testing=log/model.ckpt-19999  --log_dir=log --test_dir=../SegNet-Tutorial/CamVid/test.txt --batch_size=5 --save_image=True

# infer images
#python main.py --infer=log/model.ckpt-19999  --log_dir=log --outpath=log/0415_left_mask --test_dir=/home/dijiang/Work/data/bag_extracted/image/left/20170415/test.txt --save_image=True
