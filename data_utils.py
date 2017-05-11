#!/usr/bin/python

import sys
import os
import random
import argparse

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge

import config

bridge = CvBridge()

def preprocess_road_label(path):
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
    sample_subpath = '/train_1/' 
    sample_path = path + sample_subpath 
    label_subpath = '/label_proc_1/' 
    train_filename = path + '/train.txt'
    valid_filename = path + '/val.txt'
    train_outfile = open(train_filename, 'w')
    valid_outfile = open(valid_filename, 'w')

    for fname in os.listdir(sample_path):
        train_img = sample_subpath + fname
        name, ext = os.path.splitext(fname)
        #pre, idx = name.split('_')
        #fname_label = '_'.join([pre, mode, idx, 'proc']) + ext
        fname_label = '_'.join([name, 'proc']) + ext
        train_label = label_subpath + fname_label
        
        if random.random() < 0.05:
            valid_outfile.write(' '.join([train_img, train_label]) + '\n')
        else:
            train_outfile.write(' '.join([train_img, train_label]) + '\n')

    train_outfile.close()
    valid_outfile.close()

def calc_class_weight(path, mode='road'):
    class_map = {'p':0, 'n':0}
    for fname in os.listdir(path):
        if 'proc' in fname:
            #print(os.path.join(path, fname))
            img = cv2.imread(os.path.join(path, fname))
            #print(img.shape)
            class_map['p'] += (img>0).sum() / 3
            class_map['n'] += (img==0).sum() / 3
            #for i in range(img.shape[0]):
                #for j in range(img.shape[1]):
                    #lbl = img[i][j][0]
                    #if lbl in class_map:
                        #class_map[lbl] += 1
                    #else:
                        #class_map[lbl] = 1

    print(class_map)

def resize_image(path):
    dir = os.path.dirname(path)
    fd = open(path)
    for line in fd:
        line = line.strip()
        filename = dir + '/' + line
        #print(filename)
        img = cv2.imread(filename)
        shape = img.shape
        if shape[0] != config.IMAGE_HEIGHT or shape[1] != config.IMAGE_WIDTH:
            print("resizing {}, original shape is ({}, {})".format(line, shape[0], shape[1]))
            #if type == 'sample':
            im = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), 0, 0, cv2.INTER_AREA)
            #elif type == 'label':
                #im = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), 0, 0, cv2.INTER_NEAREST)

            cv2.imwrite(filename, im)
        
def transparent_mask(path_mask, path_raw, path_out):
    if not os.path.isdir(path_out):
        os.mkdir(path_out)
    for fname in os.listdir(path_mask):
        mask_name = path_mask + '/' + fname
        raw_name = path_raw + '/' + fname
        out_name = path_out + '/' + fname
        mask = cv2.imread(mask_name)
        raw = cv2.imread(raw_name)
        #overlay = raw.copy()
        #for i in range(overlay.shape[0]):
            #for j in range(overlay.shape[1]):
                #if mask[i][j][0] == 0 and mask[i][j][1] == 0 and mask[i][j][2] == 0:
                    #continue
                #overlay[i][j][0] = mask[i][j][0]
                #overlay[i][j][1] = mask[i][j][1]
                #overlay[i][j][2] = mask[i][j][2]
        out = cv2.addWeighted(raw, 0.7, mask, 0.3, 0)
        cv2.imwrite(out_name, out)

def _msg_to_png_file(msg, path, flip):
    img = bridge.imgmsg_to_cv2(msg)
    if flip:
        img = cv2.flip(img, -1)
    cv2.imwrite(path, img)

def _msg_compressed_to_png_file(msg, path, flip):
    img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    if flip:
        img = cv2.flip(img, -1)

    cv2.imwrite(path, img)

def _msg_show(msg, topic):
    img = bridge.imgmsg_to_cv2(msg)
    win_name = topic.split('/')[-1]
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(1)

def _msg_compressed_show(msg, topic):
    img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    win_name = topic.split('/')[1]
    if 'left' in win_name:
        img = cv2.flip(img, -1)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(1)

def rosbag_image_show(bag, topic):
    msgs = bag.read_messages(topics=[topic])
    for m in msgs:
        if m.topic == topic:
            if "/compressed" in topic:
                #if it is a compressed image topic
                img = _msg_compressed_show(m.message, topic)
            else:
                img = _msg_show(m.message, topic)

def rosbag_image_save(bag, topic, output_dir, rate=0.1):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    start_time = None
    msgs = bag.read_messages(topics=[topic])
    flip = False if 'roof' in topic else True
    for m in msgs:
        if m.topic == topic:
            if start_time is None:
                start_time = m.timestamp
            frame_number = int(((m.timestamp - start_time).to_sec() + (rate / 2.0)) / rate)
            print frame_number
            if "/compressed" in topic:
                #if it is a compressed image topic
                _msg_compressed_to_png_file(m.message, "%s/%04d.png" % (output_dir, frame_number), flip)
            else:
                _msg_to_png_file(m.message, "%s/%04d.png" % (output_dir, frame_number), flip)

def augment_image(inpath, maskpath):
    for fname in os.listdir(inpath):
        name, ext = os.path.splitext(fname)
        filename = inpath + '/' + fname
        maskname = maskpath + '/' + fname

        img = cv2.imread(filename)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        #cvt = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
        #cv2.imwrite(dir + '/' + name + '_hls' + ext, cvt)
        for i in range(-2, 3):
            if i == 0:
                continue
            tmp = hls.copy()
            tmp[:,:,0] = (tmp[:,:,0] + i*5) % 180
            cvt = cv2.cvtColor(tmp, cv2.COLOR_HLS2RGB)
            outname = name + '_hls_h' + str(i*5) + ext
            cv2.imwrite(inpath + '/' + outname, cvt)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))

            tmp = hls.copy()
            tmp[:,:,1] = (tmp[:,:,1] + i*5) % 256
            cvt = cv2.cvtColor(tmp, cv2.COLOR_HLS2RGB)
            outname = name + '_hls_l' + str(i*5) + ext
            cv2.imwrite(inpath + '/' + outname, cvt)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--func", type=str, help="functionality")
    parser.add_argument("--infile", type=str, help="input data file")
    parser.add_argument("--mode", type=str, default="road", help="lane/road")
    parser.add_argument("--inpath", type=str, help="input data path")
    parser.add_argument("--maskpath", type=str, help="input mask data path")
    parser.add_argument("--outpath", type=str, help="output data path")
    parser.add_argument("--bag", type=str, help="ros bag")
    parser.add_argument("--topic", type=str, help="ros topic")

    args = parser.parse_args()

    if args.func == 'road':
        preprocess_road_label(args.infile)
    elif args.func == 'road_split':
        split_train_valid_set(args.inpath, mode=args.mode)
    elif args.func == 'road_weight':
        calc_class_weight(args.inpath, mode=args.mode)
    elif 'resize' == args.func:
        #type = args.func.split('_')[1]
        #if type != 'sample' and type != 'label':
            #print('unknown type: {}'.format(type))
            #sys.exit(-1)
        resize_image(args.infile)
    elif args.func == 'mask':
        transparent_mask(args.maskpath, args.inpath, args.outpath)
    elif args.func == 'rosbag_img_show':
        bag = rosbag.Bag(args.bag)
        rosbag_image_show(bag, args.topic)
    elif args.func == 'rosbag_img_save':
        bag = rosbag.Bag(args.bag)
        rosbag_image_save(bag, args.topic, args.outpath)
    elif args.func == 'augment':
        augment_image(args.inpath, args.maskpath)

