#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
parser = argparse.ArgumentParser(description='estimate human pose from movie and generate VMD motion')
parser.add_argument('--reuse_2d', action='store_true', help='reuse already estimated 2D pose')
parser.add_argument('--json_dir', action='store', type=str, help='json data directory')
parser.add_argument('--first_frame', action='store', type=int, default=0, help='first frame to analyze')
parser.add_argument('--last_frame', action='store', type=int, default=-1, help='last frame to analyze')
parser.add_argument('--max_people', action='store', type=int, default=1, help='maximum number of people to analyze')
parser.add_argument('--reverse_list', action='store', type=str, default='', help='list to specify reversed person')
parser.add_argument('--order_list', action='store', type=str, default='', help='list to specify person index in left-to-right order')
parser.add_argument('--order_start_frame', action='store', type=int, default=0, help='order_start_frame')
parser.add_argument('--past_depth_dir', action='store', type=str, default='', help='depth data directory')
parser.add_argument('--add_leg', action='store_true', help='add invisible legs to estimated joints')
parser.add_argument('--add_leg2', action='store_true', help='add invisible legs to estimated joints')
parser.add_argument('--no_bg', action='store_true', help='disable BG output (show skeleton only)')
parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
parser.add_argument('VIDEO_FILE')
parser.add_argument('OUTPUT_DIR')
arg = parser.parse_args()

tfpose_dir = '../tf-pose-estimation'
depth_dir = '../mannequinchallenge-vmd'
pose3d_dir = '../3d-pose-baseline-vmd'
vmd3d_dir = '../VMD-3d-pose-baseline-multi'
rfv_dir = '../readfacevmd/build'
merge_dir = '../readfacevmd/build'
sizing_dir = '../vmd_sizing'

import logging
import datetime
import os
from pathlib import Path
from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)
if arg.verbose:
    verbose=3
    logger.setLevel(logging.DEBUG)
else:
    verbose=2
    logger.setLevel(logging.INFO)

input_video = Path(arg.VIDEO_FILE).resolve()
input_video_dir = input_video.parent
input_video_filename = input_video.stem

dttm = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

output_dir = Path(arg.OUTPUT_DIR).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

output_json_dir = output_dir / (input_video_filename + '_' + dttm) / (input_video_filename + '_json')
if arg.json_dir:
    output_json_dir = Path(arg.json_dir).resolve()
output_json_dir.mkdir(parents=True, exist_ok=True)

output_video_path = output_dir / (input_video_filename + '_' + dttm) / (input_video_filename + '_openpose.avi')


# tf-pose-estimation
tfpose_args = ['python3', 'run_video.py',
               '--video', str(input_video),
               '--model', 'mobilenet_v2_large',
               '--write_json', str(output_json_dir),
               '--number_people_max', str(arg.max_people),
               '--frame_first', str(arg.first_frame),
               '--write_video', str(output_video_path),
               '--tensorrt', 'True',
               '--no_display']
if arg.add_leg:
    tfpose_args.append('--add_leg')
if arg.no_bg:
    tfpose_args.append('--no_bg')
logger.debug('tfpose_args:' + str(tfpose_args))

if arg.reuse_2d:
    logger.info('skip 2d pose estimation')
else:
    try:
        with Popen(tfpose_args, cwd=tfpose_dir, stdout=PIPE) as proc:
            if proc.stdout:
                logger.info(proc.stdout.read())
    except CalledProcessError as e:
        print(e)
        sys.exit(1)

# update dttm
dttm = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# mannequinchallenge-vmd
depth_args = ['python3', 'predict_video.py',
             '--video_path', str(input_video),
             '--json_path', str(output_json_dir),
             '--past_depth_path', arg.past_depth_dir,
             '--interval',  '20',
             '--reverse_specific', arg.reverse_list,
             '--order_specific', arg.order_list,
             '--avi_output', 'yes',
             '--verbose',  str(verbose),
             '--number_people_max', str(arg.max_people),
             '--end_frame_no', str(arg.last_frame),
             '--now', dttm,
             '--input', 'single_view',
             '--batchSize', '1',
             '--order_start_frame', str(arg.order_start_frame)]
logger.debug('depth_args:' + str(depth_args))
print('depth_args:' + ' '.join(depth_args))

with Popen(depth_args, cwd=depth_dir, stdout=PIPE) as proc:
    if proc.stdout:
        logger.info(proc.stdout.read())

for idx in range(1, arg.max_people + 1):
    output_sub_dir = Path(str(output_json_dir) + '_' + dttm + '_idx0' + str(idx))

    # 3d-pose-baseline-vmd
    pose3d_args = ['python3', 'src/openpose_3dpose_sandbox_vmd.py',
                  '--camera_frame', '--residual', '--batch_norm',
                  '--dropout', '0.5',
                  '--max_norm', '--evaluateActionWise', '--use_sh',
                  '--epochs', '200',
                  '--load', '4874200',
                  '--gif_fps', '30',
                  '--verbose', str(verbose),
                  '--openpose', str(output_sub_dir),
                   '--person_idx', '1']
    if arg.add_leg2:
        pose3d_args.append('--add_leg')
    logger.debug('pose3d_args:' + str(pose3d_args))
    with Popen(pose3d_args, cwd=pose3d_dir, stdout=PIPE) as proc:
        if proc.stdout:
            logger.info(proc.stdout.read())

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
    with Popen(vmd3d_args, cwd=vmd3d_dir, stdout=PIPE) as proc:
        if proc.stdout:
            logger.info(proc.stdout.read())

rfv_output_dir = Path(str(output_json_dir) + '_' + dttm + '_idx01')
face_vmd_file = rfv_output_dir / (input_video_filename + '-face.vmd')
rfv_args = ['./readfacevmd',
            '--nameconf', '../nameconf.txt',
            str(input_video), str(face_vmd_file)]
logger.debug('rfv_args:' + str(rfv_args))
with Popen(rfv_args, cwd=rfv_dir, stdout=PIPE) as proc:
    if proc.stdout:
        logger.info(proc.stdout.read())

body_vmd_file = list(rfv_output_dir.glob('**/*_reduce.vmd'))[0].resolve()
merged_vmd_file = rfv_output_dir / (input_video_filename + '-merged.vmd')
merge_args = ['./mergevmd', str(body_vmd_file), str(face_vmd_file), str(merged_vmd_file)]
logger.debug('merge_args:' + str(merge_args))
with Popen(merge_args, cwd=merge_dir, stdout=PIPE) as proc:
    if proc.stdout:
        logger.info(proc.stdout.read())

target_pmx = ['data/Kaori_Skirt5.pmx', 'data/Akirose.pmx']
for pmx in target_pmx:
    sizing_args = ['python3', 'src/main.py',
                   '--vmd_path', str(merged_vmd_file),
                   '--trace_pmx_path', 'data/autotrace.pmx',
                   '--replace_pmx_path', pmx,
                   '--avoidance', '1',
                   '--hand_ik', '1',
                   '--hand_distance', '1.7',
                   '--verbose', '2']
    logger.debug('sizing_args:' + str(sizing_args))
    with Popen(sizing_args, cwd=sizing_dir, stdout=PIPE) as proc:
        if proc.stdout:
            logger.info(proc.stdout.read())
