#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# modified by Jia Xu from Copyright (c) 2019 Iván de Paz Centeno

import cv2
from mtcnn import MTCNN
import argparse
import logging

import cv2
import os


def find_images(path, recursive=False, ignore=True):
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        assert os.path.isdir(path), 'FileIO - get_images: Directory does not exist'
        assert isinstance(recursive, bool), 'FileIO - get_images: recursive must be a boolean variable'
        ext, result = ['png', 'jpg', 'jpeg'], []
        for path_a in os.listdir(path):
            path_a = os.path.join(path, path_a)
            if os.path.isdir(path_a) and recursive:
                for path_b in find_images(path_a):
                    yield path_b
            check_a = path_a.split('.')[-1] in ext
            check_b = ignore or ('-' not in path_a.split('/')[-1])
            if check_a and check_b:
                yield path_a
    else:
        raise ValueError('error! path is not a valid path or directory')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image_paths', type=str, nargs='+',  help="paths to one or more images or image directories")
    parser.add_argument('-b', '--debug', dest='debug', action='store_true', help='enable debug logging')
    parser.add_argument('-q', '--quite', dest='quite', action='store_true', help='disable all logging')
    parser.add_argument('-d', '--display', dest='display', action='store_true', help="display result")
    parser.add_argument('-s', '--save', dest='save', action='store_true', help="save result to file")
    parser.add_argument('-t', '--thresh', dest='thresh', default=0.5, type=float, help='threshold for skin mask')
    parser.add_argument('-n', '--count_num', dest='count_num', default=10, type=int, help='The number of test images')
    parser.add_argument('--save_paths', type=str, default='../bounded_skin_data/imgs',
                        help="paths to one or more images or image directories")
    args = parser.parse_args()

    detector = MTCNN()

    ###
    count_now = 0
    if not os.path.exists(args.save_paths): os.makedirs(args.save_paths)

    for image_arg in args.image_paths:
        for image_path in find_images(image_arg):
            count_now = count_now + 1
            #if count_now > args.count_num:
             #   break
            print(image_path, count_now)

            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            result = detector.detect_faces(image)
            #print("result", result)
            if len(result) == 0:
                cropImg = image

            else:

                # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
                bounding_box = result[0]['box']
                keypoints = result[0]['keypoints']
                ##(lefttop),(+w,+h)
                cv2.rectangle(image,
                              (bounding_box[0], bounding_box[1]),
                              (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                              (0, 155, 255),
                              2)
                cropImg = image[bounding_box[1]:(bounding_box[1] + bounding_box[3]),
                          bounding_box[0]:(bounding_box[0] + bounding_box[2])]  # 
				
                ### draw face feature localization
                #cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
                #cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
                #cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
                #cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
                #cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

                #cv2.imwrite("20_drawn.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            image_name = image_path.strip().split("/")[-1]
            crop_name = image_name
            crop_path = os.path.join(args.save_paths, crop_name)
            cv2.imwrite(crop_path,cv2.cvtColor(cropImg, cv2.COLOR_RGB2BGR))

