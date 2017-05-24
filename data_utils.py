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
        print(line)
        #name, ext = os.path.splitext(line)
        #outname = name + '_proc' + ext
        filename = dir + '/' + line
        #outfilename = dir + '/' + outname

        img = cv2.imread(filename)
        shape = (img.shape[0], img.shape[1], 1)
        img_proc = np.zeros(shape, np.uint8)
        #img_proc[img[:,:,0] > 0] = [1]
        for i in range(shape[0]):
            for j in range(shape[1]):
                if img[i][j][0] > 0:
                    img_proc[i][j][0] = 1
                        
        cv2.imwrite(filename, img_proc)

def split_train_valid_set(path, mode='road'):
    sample_subpath = '/sample/' 
    sample_path = path + sample_subpath 
    label_subpath = '/label/' 
    train_filename = path + '/train.txt'
    valid_filename = path + '/val.txt'
    train_outfile = open(train_filename, 'w')
    valid_outfile = open(valid_filename, 'w')

    for fname in os.listdir(sample_path):
        train_img = sample_subpath + fname
        #name, ext = os.path.splitext(fname)
        #pre, idx = name.split('_')
        #fname_label = '_'.join([pre, mode, idx, 'proc']) + ext
        #fname_label = '_'.join([name, 'proc']) + ext
        train_label = label_subpath + fname
        
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
        overlay = raw.copy()
        overlay[(mask>0)] = mask[(mask>0)]
        #for i in range(overlay.shape[0]):
            #for j in range(overlay.shape[1]):
                #if mask[i][j][0] == 0 and mask[i][j][1] == 0 and mask[i][j][2] == 0:
                    #continue
                #overlay[i][j][0] = mask[i][j][0]
                #overlay[i][j][1] = mask[i][j][1]
                #overlay[i][j][2] = mask[i][j][2]
        out = cv2.addWeighted(raw, 0.5, overlay, 0.5, 0)
        cv2.imwrite(out_name, out)

def prob_mask(path_mask, path_raw, path_out):
    if not os.path.isdir(path_out):
        os.mkdir(path_out)
    for fname in os.listdir(path_mask):
        mask_name = path_mask + '/' + fname
        raw_name = path_raw + '/' + fname
        out_name = path_out + '/' + fname
        mask = cv2.imread(mask_name)
        raw = cv2.imread(raw_name)
        #overlay = raw.copy()
        #overlay[(mask>0)] = mask[(mask>0)]
        #for i in range(overlay.shape[0]):
            #for j in range(overlay.shape[1]):
                #if mask[i][j][0] == 0 and mask[i][j][1] == 0 and mask[i][j][2] == 0:
                    #continue
                #overlay[i][j][0] = mask[i][j][0]
                #overlay[i][j][1] = mask[i][j][1]
                #overlay[i][j][2] = mask[i][j][2]
        out = cv2.addWeighted(raw, 1.0, mask, 1.0, 0)
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

def rosbag_image_save(bag, topic, output_dir, prefix='', rate=0.1):
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
                _msg_compressed_to_png_file(m.message, "%s/%s%04d.png" % (output_dir, prefix, frame_number), flip)
            else:
                _msg_to_png_file(m.message, "%s/%s%04d.png" % (output_dir, prefix, frame_number), flip)

def histEqualize(inpath, outpath):
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    for fname in os.listdir(inpath):
        print(fname)
        filename = inpath + '/' + fname
        img = cv2.imread(filename)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        cvt = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        outname = outpath + '/' + fname
        cv2.imwrite(outname, cvt)

def _add_random_shadow_v(image, side, shadow_rate):
    height = image.shape[0]
    width = image.shape[1]
    margin = 0.3 * width
    top_y = width*np.random.uniform()
    top_x = 0
    bot_x = height
    if side == 'left':
        bot_y = margin + (width - margin)*np.random.uniform()
    elif side == 'right':
        bot_y = (width - margin)*np.random.uniform()

    image_hls = image.copy()
    shadow_mask = 0*image_hls[:,:,1]
    X_m, Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1

    if side == 'left':
        cond = shadow_mask==1
    elif side == 'right':
        cond = shadow_mask==0

    image_hls = image_hls.astype(np.float64)
    image_hls[:,:,1][cond] = image_hls[:,:,1][cond]*shadow_rate
    image_hls[:,:,1] = np.where(image_hls[:,:,1] > 255, 255, image_hls[:,:,1])
    image_hls[:,:,1] = np.where(image_hls[:,:,1] < 0, 0, image_hls[:,:,1])
    image_hls = image_hls.astype(np.uint8)
    out = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return out

def _add_random_shadow_h(image, shadow_rate):
    height = image.shape[0]
    width = image.shape[1]
    margin = 0.5 * height
    left_y = 0
    left_x = margin + (height - margin)*np.random.uniform()
    right_x = margin + (height - margin)*np.random.uniform()
    right_y = width

    image_hls = image.copy()
    shadow_mask = 0*image_hls[:,:,1]
    X_m, Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]]
    shadow_mask[((X_m-left_x)*(right_y-left_y) -(right_x - left_x)*(Y_m-left_y) >=0)]=1

    cond = shadow_mask==1

    image_hls = image_hls.astype(np.float64)
    image_hls[:,:,1][cond] = image_hls[:,:,1][cond]*shadow_rate
    image_hls[:,:,1] = np.where(image_hls[:,:,1] > 255, 255, image_hls[:,:,1])
    image_hls[:,:,1] = np.where(image_hls[:,:,1] < 0, 0, image_hls[:,:,1])
    image_hls = image_hls.astype(np.uint8)
    out = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return out

def _add_random_shadow_c(image, shadow_rate):
    height = image.shape[0]
    width = image.shape[1]
    margin = 0.75 * height
    left_y = 0
    left_x1 = margin + (height - margin)*np.random.uniform()
    left_x2 = margin + (height - margin)*np.random.uniform()
    right_x1 = margin + (height - margin)*np.random.uniform()
    right_x2 = margin + (height - margin)*np.random.uniform()
    right_y = width

    top_left_x = min(left_x1, left_x2)
    bot_left_x = max(left_x1, left_x2)
    top_right_x = min(right_x1, right_x2)
    bot_right_x = max(right_x1, right_x2)

    image_hls = image.copy()
    shadow_mask = 0*image_hls[:,:,1]
    X_m, Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]]
    shadow_mask[np.logical_and((X_m-top_left_x)*(right_y-left_y) -(top_right_x - top_left_x)*(Y_m-left_y) >=0,
                               (X_m-bot_left_x)*(right_y-left_y) -(bot_right_x - bot_left_x)*(Y_m-left_y) <=0)]=1

    cond = shadow_mask==1

    image_hls = image_hls.astype(np.float64)
    image_hls[:,:,1][cond] = image_hls[:,:,1][cond]*shadow_rate
    image_hls[:,:,1] = np.where(image_hls[:,:,1] > 255, 255, image_hls[:,:,1])
    image_hls[:,:,1] = np.where(image_hls[:,:,1] < 0, 0, image_hls[:,:,1])
    image_hls = image_hls.astype(np.uint8)
    out = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return out


def augment_image(inpath, maskpath):
    for fname in os.listdir(inpath):
        name, ext = os.path.splitext(fname)
        filename = inpath + '/' + fname
        maskname = maskpath + '/' + fname

        img = cv2.imread(filename)
        r_mean = img[:,:,0].mean()
        g_mean = img[:,:,1].mean()
        b_mean = img[:,:,2].mean()

        #ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        #print("histogram equalizing")
        #ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        #cvt = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        #outname = name + '_hist' + ext
        ##cvt = cvt.astype('uint8')
        #cv2.imwrite(inpath + '/' + outname, cvt)
        #outmaskname = maskpath + '/' + outname
        #os.system('cp {} {}'.format(maskname, outmaskname))

        print("augmenting Hue/Lightness/Saturation/Contrast")
        shadow_rate_list = [0.3, 0.5]
        light_rate_list = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        for rate in light_rate_list:
            tmp = hls.copy().astype(np.float64)
            tmp[:,:,1] = tmp[:,:,1] * rate
            tmp[:,:,1] = np.where(tmp[:,:,1] > 255, 255, tmp[:,:,1])
            tmp[:,:,1] = np.where(tmp[:,:,1] < 0, 0, tmp[:,:,1])
            tmp = tmp.astype(np.uint8)
            outimg = cv2.cvtColor(tmp, cv2.COLOR_HLS2RGB)
            outname = name + '_hls_l{}'.format(rate) + ext
            cv2.imwrite(inpath + '/' + outname, outimg)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))

        for i,rate in enumerate(shadow_rate_list):
            oname = name + '_shadow{}'.format(rate)
            # left
            outimg = _add_random_shadow_v(hls, 'left', rate)
            outname = oname + '_left' + ext
            cv2.imwrite(inpath + '/' + outname, outimg)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))

            # right
            outimg = _add_random_shadow_v(hls, 'right', rate)
            outname = oname + '_right' + ext
            cv2.imwrite(inpath + '/' + outname, outimg)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))

            # bottom
            outimg = _add_random_shadow_h(hls, rate)
            outname = oname + '_bottom' + ext
            cv2.imwrite(inpath + '/' + outname, outimg)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))

            # cross
            outimg = _add_random_shadow_c(hls, rate)
            outname = oname + '_cross' + ext
            cv2.imwrite(inpath + '/' + outname, outimg)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))

        for i in range(-2, 3):
            if i == 0:
                continue
            # Hue
            tmp = hls.copy().astype('int16')
            tmp[:,:,0] = (tmp[:,:,0] + i*5 + 180) % 180
            tmp = tmp.astype('uint8')
            cvt = cv2.cvtColor(tmp, cv2.COLOR_HLS2RGB)
            outname = name + '_hls_h' + str(i*5) + ext
            cv2.imwrite(inpath + '/' + outname, cvt)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))

            # Lightness
            #tmp = hls.copy().astype('int16')
            #tmp[:,:,1] += i*10
            #tmp[:,:,1] = np.where(tmp[:,:,1] > 255, 255, tmp[:,:,1])
            #tmp[:,:,1] = np.where(tmp[:,:,1] < 0, 0, tmp[:,:,1])
            #tmp = tmp.astype('uint8')
            #cvt = cv2.cvtColor(tmp, cv2.COLOR_HLS2RGB)
            #outname = name + '_hls_l' + str(i*10) + ext
            #cv2.imwrite(inpath + '/' + outname, cvt)
            #outmaskname = maskpath + '/' + outname
            #os.system('cp {} {}'.format(maskname, outmaskname))

            # Saturation
            tmp = hls.copy().astype('int16')
            tmp[:,:,2] += i*10
            tmp[:,:,2] = np.where(tmp[:,:,2] > 255, 255, tmp[:,:,2])
            tmp[:,:,2] = np.where(tmp[:,:,2] < 0, 0, tmp[:,:,2])
            tmp = tmp.astype('uint8')
            cvt = cv2.cvtColor(tmp, cv2.COLOR_HLS2RGB)
            outname = name + '_hls_s' + str(i*10) + ext
            cv2.imwrite(inpath + '/' + outname, cvt)
            outmaskname = maskpath + '/' + outname
            os.system('cp {} {}'.format(maskname, outmaskname))
            
            #tmp = hls.copy().astype('int16')
            #for c in  range(3):
                #tmp[:,:,i] += i*5
                #tmp[:,:,i] = np.where(tmp[:,:,i] > 255, 255, tmp[:,:,i])
                #tmp[:,:,i] = np.where(tmp[:,:,i] < 0, 0, tmp[:,:,i])
            #tmp = tmp.astype('uint8')
            #outname = name + '_hls_b' + str(i*5) + ext
            #cv2.imwrite(inpath + '/' + outname, tmp)
            #outmaskname = maskpath + '/' + outname
            ##os.system('cp {} {}'.format(maskname, outmaskname))

            # Contrast
            tmp = img.copy()
            factor = 1. + i * 0.3
            r = tmp[:,:,0].astype('float32')
            g = tmp[:,:,1].astype('float32')
            b = tmp[:,:,2].astype('float32')
            r = (r - r_mean) * factor + r_mean
            r = np.where(r > 255, 255, r)
            r = np.where(r < 0, 0, r)
            r = r.astype('uint8')
            g = (g - g_mean) * factor + g_mean
            g = np.where(g > 255, 255, g)
            g = np.where(g < 0, 0, g)
            g = g.astype('uint8')
            b = (b - b_mean) * factor + b_mean
            b = np.where(b > 255, 255, b)
            b = np.where(b < 0, 0, b)
            b = b.astype('uint8')
            tmp[:,:,0] = r
            tmp[:,:,1] = g
            tmp[:,:,2] = b

            outname = name + '_hls_c' + str(factor) + ext
            cv2.imwrite(inpath + '/' + outname, tmp)
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
    parser.add_argument("--prefix", default='', type=str, help="bag extrated image prefix")

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
    elif args.func == 'probmask':
        prob_mask(args.maskpath, args.inpath, args.outpath)
    elif args.func == 'rosbag_img_show':
        bag = rosbag.Bag(args.bag)
        rosbag_image_show(bag, args.topic)
    elif args.func == 'rosbag_img_save':
        bag = rosbag.Bag(args.bag)
        rosbag_image_save(bag, args.topic, args.outpath, args.prefix)
    elif args.func == 'augment':
        augment_image(args.inpath, args.maskpath)
    elif args.func == 'hist':
        histEqualize(args.inpath, args.outpath)

