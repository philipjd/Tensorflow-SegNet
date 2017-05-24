#!/bin/bash

# for training
python main.py --log_dir=log --image_dir=/home/dijiang/Work/data/lane_detection/train.txt --val_dir=/home/dijiang/Work/data/lane_detection/val.txt --batch_size=5
