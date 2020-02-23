#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2
import numpy as np

def plot_joints(image, joints, scale):
    height = image.shape[0]
    width = image.shape[1]
    bones = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
             (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
             (8, 14), (14, 15), (15, 16)]
    for i, j in bones:
        x1, y1, z1 = joints[i]
        x2, y2, z2 = joints[j]
        x1 = int((x1 * scale + 0.5) * width + 0.5)
        x2 = int((x2 * scale + 0.5) * width + 0.5)
        y1 = height - int((y1 * scale + 0.2) * height + 0.5)
        y2 = height - int((y2 * scale + 0.2) * height + 0.5)
        #print(x1, y1, x2, y2)
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return image

def calc_scale(joints):
    ymax = -1
    ymin = 1000
    for (x, y, z) in joints:
        if y > ymax:
            ymax = y
        if y < ymin:
            ymin = y

    return 0.5/(ymax - ymin)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot joints in pos.txt')
    parser.add_argument('POS_FILE')
    parser.add_argument('OUTPUT_VIDEO_FILE')
    arg = parser.parse_args()
    width = 512
    height = 384
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'IYUV')
    out = cv2.VideoWriter(arg.OUTPUT_VIDEO_FILE, fourcc, 30.0, (width, height))
    with open(arg.POS_FILE, "r") as fin:
        lines = fin.readlines()
        scale = -1
        for line in lines:
            positions = line.split(', ')[:-1]
            joints = []
            image = np.zeros((height, width, 3), np.uint8)
            for pos in positions:
                # print("pos: ", pos)
                id, x, z, y = pos.split(' ')
                # print("x, y, z = ", x, y, z)
                joints.append((float(x), float(y), float(z)))
            if scale < 0:
                scale = calc_scale(joints)
            image = plot_joints(image, joints, scale)
            out.write(image)

    out.release()
