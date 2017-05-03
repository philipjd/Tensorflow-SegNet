#!/usr/bin/python

import sys
import os
import random
import cv2
import numpy as np

def preprocess_road(path):
    dir = os.path.dirname(path)
    fd = open(path)
    for line in fd:
        line = line.strip()
        name, ext = os.path.splitext(line)
        outname = name + '_proc' + ext
        filename = dir + '/' + line
        outfilename = dir + '/' + outname

        img = cv2.imread(filename)
        shape = (img.shape[0], img.shape[1], 1)
        img_proc = np.zeros(shape, np.uint8)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if img[i][j][0] > 0:
                    img_proc[i][j][0] = 1
                        
        cv2.imwrite(outfilename, img_proc)

def split_train_valid_set(path, mode='road'):
    sample_subpath = '/image_2/' 
    sample_path = path + sample_subpath 
    label_subpath = '/gt_image_2/' 
    train_filename = path + '/train.txt'
    valid_filename = path + '/val.txt'
    train_outfile = open(train_filename, 'w')
    valid_outfile = open(valid_filename, 'w')

    for fname in os.listdir(sample_path):
        train_img = sample_subpath + fname
        name, ext = os.path.splitext(fname)
        pre, idx = name.split('_')
        fname_label = '_'.join([pre, mode, idx, 'proc']) + ext
        train_label = label_subpath + fname_label
        
        if random.random() < 0.2:
            valid_outfile.write(' '.join([train_img, train_label]) + '\n')
        else:
            train_outfile.write(' '.join([train_img, train_label]) + '\n')

    train_outfile.close()
    valid_outfile.close()

def calc_class_weight(path, mode='road'):
    class_map = {}
    for fname in os.listdir(path):
        if mode in fname and 'proc' in fname:
            #print(os.path.join(path, fname))
            img = cv2.imread(os.path.join(path, fname))
            #print(img.shape)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    lbl = img[i][j][0]
                    if lbl in class_map:
                        class_map[lbl] += 1
                    else:
                        class_map[lbl] = 1

    print(class_map)




if __name__ == '__main__':
    if sys.argv[1] == 'road':
        preprocess_road(sys.argv[2])
    elif sys.argv[1] == 'road_split':
        split_train_valid_set(sys.argv[2], 'road')
    elif sys.argv[1] == 'road_weight':
        calc_class_weight(sys.argv[2], mode='road')

