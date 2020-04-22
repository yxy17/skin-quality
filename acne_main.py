#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'

import argparse
import logging

import cv2
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure, color




logger = logging.getLogger('main')

def find_images(path, recursive=False, ignore=True):
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        assert os.path.isdir(path), 'FileIO - get_images: Directory does not exist'
        assert isinstance(recursive, bool), 'FileIO - get_images: recursive must be a boolean variable'
        ext, result = ['png', 'jpg', 'jpeg'], []
        for path_a in os.listdir(path):
            path_a = os.path.join(path , path_a)
            if os.path.isdir(path_a) and recursive:
                for path_b in find_images(path_a):
                    yield path_b
            check_a = path_a.split('.')[-1] in ext
            check_b = ignore or ('-' not in path_a.split('/')[-1])
            if check_a and check_b:
                yield path_a
    else:
        raise ValueError('error! path is not a valid path or directory')


## 0 黑色  ##255白色
def canny_edge(image_path, thr1, thr2, save_path):
    img = cv2.imread(image_path, 0)
    canny = cv2.Canny(img, thr1, thr2)

    image_name = image_path.strip().split("/")[-1]
    mask_name = "msk_" + image_name
    mask_path = os.path.join(save_path, mask_name)
    cv2.imwrite(mask_path, img_msk)

    #plt.imshow(canny)
    #plt.show()
    return canny

def get_skin_mask(image_path,save_path):
    #imname = "acne2.png"

    '''
    YCrCb  Cr   + OTSU
    '''
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    (y, cr, cb) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    image_name = image_path.strip().split("\\")[-1]
    mask_name = "msk_" + image_name
    mask_path = os.path.join(save_path, mask_name)
    cv2.imwrite(mask_path, skin)

    #return img_msk

    return skin


def RobertsOperator(roi):
    operator_first = np.array([[-1, 0], [0, 1]])
    operator_second = np.array([[0, -1], [1, 0]])
    return np.abs(np.sum(roi[1:, 1:] * operator_first)) + np.abs(np.sum(roi[1:, 1:] * operator_second))


def RobertsAlogrithm(image):
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            image[i, j] = RobertsOperator(image[i - 1:i + 2, j - 1:j + 2])
    return image[1:(image.shape[0] - 1), 1:(image.shape[1] - 1)]


def obtain_edge(img_path, save_path, thres=20):
    saber = cv2.imread(img_path)
    print("saber", saber.shape)
    saber = cv2.cvtColor(saber, cv2.COLOR_BGR2RGB)
    # plt.imshow(saber)
    # plt.axis("off")
    # plt.show()
    gray_saber = cv2.cvtColor(saber, cv2.COLOR_RGB2GRAY)
    # gray_saber = cv2.resize(gray_saber,(200,200))
    # print("gray_saber",gray_saber.shape)
    Robert_saber = RobertsAlogrithm(gray_saber)
    _, binart_img = cv2.threshold(Robert_saber, thres, 255, cv2.THRESH_BINARY)
    # plt.imshow(binart_img)
    # plt.axis("off")
    # plt.show()
    image_name = image_path.strip().split("\\")[-1]
    mask_name = "edge_" + image_name
    mask_path = os.path.join(save_path, mask_name)
    print("mask_path",mask_path)
    cv2.imwrite(mask_path, binart_img)
    return binart_img

def list_to_txt(list1,out_file):
    ## 需要[x_leftop, y_lefttop, x_rightbottm, y_rightbottom,(x0,y0,x1,y1)
    ##输入是（y0,x0,y1,x1)
    f_o = open(out_file, 'w')
    for item in list1:
        [width, top, height, left] = [item[1],item[0],item[3],item[2]]
        f_o.write('{} {} {} {}\n'.format(left, top, width, height))
    f_o.close()

def combine_bi_skin_img(img, mask,save_path,image_path):
    # path = 'msk_acne2.png'
    # img = cv2.imread(mask_path)
    # img = cv2.resize(img,(200,200))
    # img = cv2.copyMakeBorder(img,1,0,1,0,cv2.BORDER_DEFAULT)
    mask = mask / 255
    mask = mask.astype(int)
    print(mask.shape)
    # mask = cv2.imread(ori_path)
    print(img.shape)
    joino = img * mask
    #cv2.imwrite("join.png", joino)
    #plt.imshow(joino)
    #plt.show()

    labels_all = measure.label(joino, connectivity=2, return_num=True)
    labels = labels_all[0]
    # print("type(label)",type(labels),labels.shape,type(labels[1,1]))
    # print(labels)
    # print(np.unique(labels))
    dst = color.label2rgb(labels.astype(int))
    #plt.imshow(dst)
    #plt.show()

    props = measure.regionprops(labels)
    bb_cneter = []
    bb = []
    for i in range(labels_all[1]):
        bb_cneter.append(props[i].centroid)
        bb.append(props[i].bbox)
    ###(min_row, min_col, max_row, max_col)--->（y0,x0,y1,x1)
    print("bb",bb,len(bb))
    print(type(bb[0]),type(bb))
    image_name = image_path.strip().split("\\")[-1]
    txt_name = image_name.split(".jpg")[0]+".txt"
    txt_path = os.path.join(save_path,txt_name)
    list_to_txt(bb, txt_path)

    return bb, bb_cneter




def combine_bi_skin(ori_path, mask_path):
    # path = 'msk_acne2.png'
    mask = cv2.imread(mask_path,0)
    #img = cv2.resize(img,(200,200))
    # img = cv2.copyMakeBorder(img,1,0,1,0,cv2.BORDER_DEFAULT)
    mask = mask / 255
    mask = mask.astype(int)
    print(mask.shape)
    img = cv2.imread(ori_path)
    print(img.shape)
    joino = img * mask
    cv2.imwrite("join.png", joino)

    labels_all = measure.label(joino, connectivity=2, return_num=True)
    labels = labels_all[0]

    dst = color.label2rgb(labels.astype(int))
    #plt.imshow(dst)
    #plt.show()

    props = measure.regionprops(labels)
    bb_cneter = []
    bb = []
    for i in range(labels_all[1]):
        bb_cneter.append(props[i].centroid)
        bb.append(props[i].bbox)

    print("bb",bb.shape,type(bb))
    return bb, bb_cneter




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image_paths', type=str, default=['./eval_img/pimple_test_img'],nargs='+', help="paths to one or more images or image directories")
    parser.add_argument('--thresh', dest='thresh', default=35, type=float, help='threshold for skin mask')
    parser.add_argument('-n', '--count_num', dest='count_num', default=None, type=int, help='The number of test images')
    parser.add_argument('--save_path', type=str, default='./bbox_output', help="paths to one or more images or image directories")
    parser.add_argument('--mask_save_path', type=str, default='./mask', help="paths to one or more images or image directories")
    parser.add_argument('--edge_save_path', type=str, default='./edge', help="paths to one or more images or image directories")
    #parser.add_argument('--edge_save_path', type=str, default='./edge', help="paths to one or more images or image directories")

    args = parser.parse_args()


    logger = logging.getLogger("main")

    ### for counting number
    count_now = 0
    ##  for saving
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    if not os.path.exists(args.mask_save_path): os.makedirs(args.mask_save_path)
    if not os.path.exists(args.edge_save_path): os.makedirs(args.edge_save_path)
    save_path = args.save_path
    mask_save_path = args.mask_save_path
    edge_save_path = args.edge_save_path

    thres = args.thresh
    for image_arg in args.image_paths:
        for image_path in find_images(image_arg):
            ## for part of images
            count_now = count_now + 1
            if args.count_num != None:
                if count_now > args.count_num:
                    break
            print(image_path, count_now)

            logging.info("loading image from {0}".format(image_path))
            # 906,786,3
            img_msk = get_skin_mask(image_path, mask_save_path)
            print(" img_msk ", img_msk.shape)  # 1280,720
            binart_img = obtain_edge(image_path, edge_save_path, thres)  # 1280,720
            print("binart_img ", binart_img.shape)
            bb, bb_cneter = combine_bi_skin_img(binart_img, img_msk,save_path,image_path)






