#!/bin/bash

source /home/icub/.virtualenvs/mrcnn/bin/activate

python online_demo_yarp.py --width 320 --height 240 $ROBOT_CODE/icub-contrib-iit/Mask_RCNN/models/mask_rcnn_ycb_video_training_0040.h5
