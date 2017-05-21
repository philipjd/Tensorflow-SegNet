#!/bin/bash

# for training
#python main.py --log_dir=log --image_dir=/home/dijiang/Work/data/lane_detection/train.txt --val_dir=/home/dijiang/Work/data/lane_detection/val.txt --batch_size=5

# for finetune from saved ckpt
# python main.py --finetune=log/model.ckpt-19999  --log_dir=log --image_dir=../SegNet-Tutorial/CamVid/train.txt --val_dir=../SegNet-Tutorial/CamVid/val.txt --batch_size=5

# for testing
# python main.py --testing=log/model.ckpt-19999  --log_dir=log --test_dir=../SegNet-Tutorial/CamVid/test.txt --batch_size=5 --save_image=True

# infer images
python main.py --infer=model/plusai_v2_model.ckpt-39999  --log_dir=log --outpath=log/plusai_0430_hw1_left_mask_v2 --test_dir=/home/dijiang/Work/data/bag_extracted/image/left/20170430_highway1_resize/img.txt --save_image=True
