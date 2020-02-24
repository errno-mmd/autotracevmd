#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import datetime
import json
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

def estimate_pose2d(input_video, output_json_dir, pose2d_video, conf):
    # tf-pose-estimation
    tfpose_args = ['python3', 'run_video.py',
                '--video', str(input_video),
                '--model', 'mobilenet_v2_large',
                '--write_json', str(output_json_dir),
                '--number_people_max', str(conf['max_people']),
                '--frame_first', str(conf['first_frame']),
                '--write_video', str(pose2d_video),
                '--no_display']
    if conf['no_bg']:
        tfpose_args.append('--no_bg')
    logger.debug('tfpose_args:' + str(tfpose_args))
    subprocess.run(tfpose_args, cwd=TFPOSE_DIR, check=True)

def estimate_depth(input_video, output_json_dir, dttm, conf):
    # mannequinchallenge-vmd
    depth_args = ['python3', 'predict_video.py',
                '--video_path', str(input_video),
                '--json_path', str(output_json_dir),
                '--interval',  '20',
                '--reverse_specific', conf['reverse_list'],
                '--order_specific', conf['order_list'],
                '--avi_output', 'yes',
                '--verbose',  str(conf['log_level']),
                '--number_people_max', str(conf['max_people']),
                '--end_frame_no', str(conf['last_frame']),
                '--now', dttm,
                '--input', 'single_view',
                '--batchSize', '1',
                '--order_start_frame', str(conf['order_start_frame'])]
    logger.debug('depth_args:' + str(depth_args))
    subprocess.run(depth_args, cwd=DEPTH_DIR, check=True)

def estimate_pose3d(output_sub_dir, conf):
    # 3d-pose-baseline-vmd
    pose3d_args = ['python3', 'src/openpose_3dpose_sandbox_vmd.py',
                  '--camera_frame', '--residual', '--batch_norm',
                  '--dropout', '0.5',
                  '--max_norm', '--evaluateActionWise', '--use_sh',
                  '--epochs', '200',
                  '--load', '4874200',
                  '--gif_fps', '30',
                  '--verbose', str(conf['log_level']),
                  '--openpose', str(output_sub_dir),
                   '--person_idx', '1']
    if conf['add_leg']:
        pose3d_args.append('--add_leg')
    logger.debug('pose3d_args:' + str(pose3d_args))
    subprocess.run(pose3d_args, cwd=POSE3D_DIR, check=True)

def pose3d_to_vmd(output_sub_dir, conf):
    # VMD-3d-pose-baseline-multi
    vmd3d_args = ['python3', 'main.py',
                  '-v', '2',
                  '-t', str(output_sub_dir),
                  '-b', conf['vmd3d_bone_csv'] if 'vmd3d_bone_csv' in conf else 'born/animasa_miku_born.csv',
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

def add_face_motion(input_video, output_json_dir, input_video_filename, dttm, conf):
    rfv_output_dir = pathlib.Path(str(output_json_dir) + '_' + dttm + '_idx01')
    face_vmd_file = rfv_output_dir / (input_video_filename + '-face.vmd')
    rfv_args = ['./readfacevmd',
                str(input_video), str(face_vmd_file)]
    if 'rfv_nameconf' in conf:
        rfv_args.extend(['--nameconf', conf['rfv_nameconf']])
    logger.debug('rfv_args:' + str(rfv_args))
    subprocess.run(rfv_args, cwd=RFV_DIR, check=True)

    body_vmd_file = list(rfv_output_dir.glob('**/*_reduce.vmd'))[0].resolve()
    merged_vmd_file = rfv_output_dir / (input_video_filename + '-merged.vmd')
    merge_args = ['./mergevmd', str(body_vmd_file), str(face_vmd_file), str(merged_vmd_file)]
    logger.debug('merge_args:' + str(merge_args))
    subprocess.run(merge_args, cwd=RFV_DIR, check=True)
    return merged_vmd_file

def resize_motion(src_vmd_file, trace_pmx, replace_pmx, conf):
    sizing_args = ['python3', 'src/main.py',
                   '--vmd_path', str(src_vmd_file),
                   '--trace_pmx_path', trace_pmx,
                   '--replace_pmx_path', replace_pmx,
                   '--avoidance', '1',
                   '--hand_ik', '1',
                   '--hand_distance', '1.7',
                   '--verbose', str(conf['log_level'])]
    logger.debug('sizing_args:' + str(sizing_args))
    subprocess.run(sizing_args, cwd=SIZING_DIR, check=True)

if __name__ == '__main__':
    config_file = pathlib.Path('config.json')
    if config_file.is_file():
        with config_file.open() as fconf:
            conf = json.load(fconf)
    else:
        conf = {}
    parser = argparse.ArgumentParser(description='estimate human pose from movie and generate VMD motion')
    parser.add_argument('--output_dir', action='store', type=str, default=conf['output_dir'] if 'output_dir' in conf else '', help='output directory')
    parser.add_argument('--log_level', action='store', type=int, default=conf['output_dirlog_level'] if 'log_level' in conf else 1, help='log verbosity')
    parser.add_argument('--first_frame', action='store', type=int, default=conf['first_frame'] if 'first_frame' in conf else 0, help='first frame to analyze')
    parser.add_argument('--last_frame', action='store', type=int, default=conf['last_frame'] if 'last_frame' in conf else -1, help='last frame to analyze')
    parser.add_argument('--max_people', action='store', type=int, default=conf['max_people'] if 'max_people' in conf else 1, help='maximum number of people to analyze')
    parser.add_argument('--reverse_list', action='store', type=str, default=conf['reverse_list'] if 'reverse_list' in conf else '', help='list to specify reversed person')
    parser.add_argument('--order_list', action='store', type=str, default=conf['order_list'] if 'order_list' in conf else '', help='list to specify person index in left-to-right order')
    parser.add_argument('--order_start_frame', action='store', type=int, default=conf['order_start_frame'] if 'order_start_frame' in conf else 0, help='order_start_frame')
    parser.add_argument('--add_leg', action='store_true', default=conf['add_leg'] if 'add_leg' in conf else False, help='add invisible legs to estimated joints')
    parser.add_argument('--no_bg', action='store_true', default=conf['no_bg'] if 'no_bg' in conf else False, help='disable BG output (show skeleton only)')
    parser.add_argument('VIDEO_FILE')
    arg = parser.parse_args()
    argdic = vars(arg)
    for k in argdic.keys():
        conf[k] = argdic[k]

    logger = logging.getLogger(__name__)
    if conf['log_level'] == 0:
        logger.setLevel(logging.ERROR)
    elif conf['log_level'] == 1:
        logger.setLevel(logging.WARNING)
    elif conf['log_level'] == 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    input_video = pathlib.Path(conf['VIDEO_FILE']).resolve()
    input_video_dir = input_video.parent
    input_video_filename = input_video.stem
    output_dir = input_video_dir
    if conf['output_dir'] != '':
        output_dir = pathlib.Path(conf['output_dir']).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dttm = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_json_dir = output_dir / (input_video_filename + '_' + dttm) / (input_video_filename + '_json')
    output_json_dir.mkdir(parents=True, exist_ok=True)
    pose2d_video = output_dir / (input_video_filename + '_' + dttm) / (input_video_filename + '_openpose.avi')

    estimate_pose2d(input_video, output_json_dir, pose2d_video, conf)
    # update dttm
    dttm = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    estimate_depth(input_video, output_json_dir, dttm, conf)
    for idx in range(1, conf['max_people'] + 1):
        output_sub_dir = pathlib.Path(str(output_json_dir) + '_' + dttm + '_idx0' + str(idx))
        estimate_pose3d(output_sub_dir, conf)
        pose3d_to_vmd(output_sub_dir, conf)

    if 'rfv_enable' in conf and conf['rfv_enable']:
        sizing_src_vmd = add_face_motion(input_video, output_json_dir, input_video_filename, dttm, conf)
    else:
        vmd_output_dir = pathlib.Path(str(output_json_dir) + '_' + dttm + '_idx01')
        sizing_src_vmd = list(vmd_output_dir.glob('**/*_reduce.vmd'))[0].resolve()

    if 'sizing_trace_pmx' in conf and 'sizing_replace_pmx_list' in conf:
      for replace_pmx in conf['sizing_replace_pmx_list']:
            resize_motion(sizing_src_vmd, conf['sizing_trace_pmx'], replace_pmx, conf)
