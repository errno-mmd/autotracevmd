#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import datetime
import os
import pathlib
import subprocess
import sys

TFPOSE_DIR = '../tf-pose-estimation'
DEPTH_DIR = '../mannequinchallenge-vmd'
POSE3D_DIR = '../3d-pose-baseline-vmd'
VMD3D_DIR = '../VMD-3d-pose-baseline-multi'
RFV_DIR = '../readfacevmd/build'
SIZING_DIR = '../vmd_sizing'

def estimate_pose2d(input_video, output_json_dir, pose2d_video, arg):
    # tf-pose-estimation
    tfpose_args = ['python3', 'run_video.py',
                '--video', str(input_video),
                '--model', 'mobilenet_v2_large',
                '--write_json', str(output_json_dir),
                '--number_people_max', str(arg.max_people),
                '--frame_first', str(arg.first_frame),
                '--write_video', str(pose2d_video),
                '--no_display']
    if arg.no_bg:
        tfpose_args.append('--no_bg')
    logger.debug('tfpose_args:' + str(tfpose_args))
    subprocess.run(tfpose_args, cwd=TFPOSE_DIR, check=True)

def estimate_depth(input_video, output_json_dir, dttm, arg):
    # mannequinchallenge-vmd
    depth_args = ['python3', 'predict_video.py',
                '--video_path', str(input_video),
                '--json_path', str(output_json_dir),
                '--interval',  '20',
                '--reverse_specific', arg.reverse_list,
                '--order_specific', arg.order_list,
                '--avi_output', 'yes',
                '--verbose',  str(arg.log_level),
                '--number_people_max', str(arg.max_people),
                '--end_frame_no', str(arg.last_frame),
                '--now', dttm,
                '--input', 'single_view',
                '--batchSize', '1',
                '--order_start_frame', str(arg.order_start_frame)]
    logger.debug('depth_args:' + str(depth_args))
    print('depth_args:' + ' '.join(depth_args))
    subprocess.run(depth_args, cwd=DEPTH_DIR, check=True)

def estimate_pose3d(output_sub_dir, arg):
    # 3d-pose-baseline-vmd
    pose3d_args = ['python3', 'src/openpose_3dpose_sandbox_vmd.py',
                  '--camera_frame', '--residual', '--batch_norm',
                  '--dropout', '0.5',
                  '--max_norm', '--evaluateActionWise', '--use_sh',
                  '--epochs', '200',
                  '--load', '4874200',
                  '--gif_fps', '30',
                  '--verbose', str(arg.log_level),
                  '--openpose', str(output_sub_dir),
                   '--person_idx', '1']
    if arg.add_leg:
        pose3d_args.append('--add_leg')
    logger.debug('pose3d_args:' + str(pose3d_args))
    subprocess.run(pose3d_args, cwd=POSE3D_DIR, check=True)

def pose3d_to_vmd(output_sub_dir, arg):
    # VMD-3d-pose-baseline-multi
    vmd3d_args = ['python3', 'main.py',
                  '-v', '2',
                  '-t', str(output_sub_dir),
#                  '-b', 'born/animasa_miku_born.csv',
                  '-b', 'born/autotrace_bone.csv',
                  '-c', '30',
                  '-z', '1.5',
                  '-s', '1',
                  '-p', '0.5',
                  '-r', '5',
                  '-k', '1',
                  '-e', '0',
                  '-d', '4']
    logger.debug('vmd3d_args:' + str(vmd3d_args))
    subprocess.run(vmd3d_args, cwd=VMD3D_DIR, check=True)

def add_face_motion(input_video, output_json_dir, input_video_filename, dttm, arg):
    rfv_output_dir = pathlib.Path(str(output_json_dir) + '_' + dttm + '_idx01')
    face_vmd_file = rfv_output_dir / (input_video_filename + '-face.vmd')
    rfv_args = ['./readfacevmd',
                '--nameconf', '../nameconf.txt',
                str(input_video), str(face_vmd_file)]
    logger.debug('rfv_args:' + str(rfv_args))
    subprocess.run(rfv_args, cwd=RFV_DIR, check=True)

    body_vmd_file = list(rfv_output_dir.glob('**/*_reduce.vmd'))[0].resolve()
    merged_vmd_file = rfv_output_dir / (input_video_filename + '-merged.vmd')
    merge_args = ['./mergevmd', str(body_vmd_file), str(face_vmd_file), str(merged_vmd_file)]
    logger.debug('merge_args:' + str(merge_args))
    subprocess.run(merge_args, cwd=RFV_DIR, check=True)
    return merged_vmd_file

def resize_motion(merged_vmd_file, pmx_file):
    sizing_args = ['python3', 'src/main.py',
                   '--vmd_path', str(merged_vmd_file),
                   '--trace_pmx_path', 'data/autotrace.pmx',
                   '--replace_pmx_path', pmx_file,
                   '--avoidance', '1',
                   '--hand_ik', '1',
                   '--hand_distance', '1.7',
                   '--verbose', str(arg.log_level)]
    logger.debug('sizing_args:' + str(sizing_args))
    subprocess.run(sizing_args, cwd=SIZING_DIR, check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='estimate human pose from movie and generate VMD motion')
    parser.add_argument('--output_dir', action='store', type=str, default='', help='output directory')
    parser.add_argument('--log_level', action='store', type=int, default=1, help='log verbosity')
    parser.add_argument('--first_frame', action='store', type=int, default=0, help='first frame to analyze')
    parser.add_argument('--last_frame', action='store', type=int, default=-1, help='last frame to analyze')
    parser.add_argument('--max_people', action='store', type=int, default=1, help='maximum number of people to analyze')
    parser.add_argument('--reverse_list', action='store', type=str, default='', help='list to specify reversed person')
    parser.add_argument('--order_list', action='store', type=str, default='', help='list to specify person index in left-to-right order')
    parser.add_argument('--order_start_frame', action='store', type=int, default=0, help='order_start_frame')
    parser.add_argument('--add_leg', action='store_true', help='add invisible legs to estimated joints')
    parser.add_argument('--no_bg', action='store_true', help='disable BG output (show skeleton only)')
    parser.add_argument('VIDEO_FILE')
    arg = parser.parse_args()

    logger = logging.getLogger(__name__)
    if arg.log_level == 0:
        logger.setLevel(logging.ERROR)
    elif arg.log_level == 1:
        logger.setLevel(logging.WARNING)
    elif arg.log_level == 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    input_video = pathlib.Path(arg.VIDEO_FILE).resolve()
    input_video_dir = input_video.parent
    input_video_filename = input_video.stem
    output_dir = input_video_dir
    if arg.output_dir != '':
        output_dir = pathlib.Path(arg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dttm = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_json_dir = output_dir / (input_video_filename + '_' + dttm) / (input_video_filename + '_json')
    output_json_dir.mkdir(parents=True, exist_ok=True)
    pose2d_video = output_dir / (input_video_filename + '_' + dttm) / (input_video_filename + '_openpose.avi')

    estimate_pose2d(input_video, output_json_dir, pose2d_video, arg)
    # update dttm
    dttm = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    estimate_depth(input_video, output_json_dir, dttm, arg)
    for idx in range(1, arg.max_people + 1):
        output_sub_dir = pathlib.Path(str(output_json_dir) + '_' + dttm + '_idx0' + str(idx))
        estimate_pose3d(output_sub_dir, arg)
        pose3d_to_vmd(output_sub_dir, arg)

    merged_vmd_file = add_face_motion(input_video, output_json_dir, input_video_filename, dttm, arg)
    target_pmx = ['data/Kaori_Skirt5.pmx', 'data/Akirose.pmx']
    for pmx_file in target_pmx:
        resize_motion(merged_vmd_file, pmx_file)
