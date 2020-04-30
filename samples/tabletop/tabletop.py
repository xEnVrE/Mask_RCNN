"""
Mask R-CNN

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Train, Detect and Evaluate on a Tabletop dataset
Adapted by Fabrizio Bottarel (fabrizio.bottarel@iit.it)

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO or imagenet weights
    python3 tabletop.py train --dataset=/path/to/dataset/root --weights=[coco, imagenet]

    # Resume training a model that you had trained earlier
    python3 tabletop.py train --dataset=/path/to/dataset/root --weights=last

    # Train a new model starting from ImageNet weights
    python3 tabletop.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Splash results to an image
    python3 tabletop.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Splash results to video using the last weights you trained
    python3 tabletop.py splash --weights=last --video=<URL or path to file>

    # Save masks of all objects given a sequence of images
    python3 tabletop.py masks --weights=/path/to/weights/file.h5 --dataset=/path/to/dataset --sequence=/path/to/sequence --output=/path/to/output --number_classes=<number of classes>
"""

import glob
import os
import sys
import datetime
import numpy as np
import cv2
import imageio
from keras.utils.generic_utils import Progbar
import imgaug

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import config and dataset files
from samples.tabletop import configurations
from samples.tabletop import datasets

# Import Mask RCNN
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Utils
############################################################

def apply_detection_results(image, masks, bboxes, class_ids, class_names, colors, scores=None):
    """
    Performs inference on target image and draws them on a return image.
    :param image (np.ndarray): 3-channel, 8-bit RGB image. Size must be according to CONFIG file (default 640x480)
    :param masks (np.ndarray): binary array of size [height, width, no_of_detections]
    :param bboxes (list): list of bboxes. Each bbox is a tuple (y1, x1, y2, x2)
    :param class_ids (list): one numerical ID for each detection
    :param class_names (list): string of class names, as given by Dataset.class_names
    :param colors (dict): keys are class names, values are float [0 : 1] 3d tuples representing RGB color
    :param scores (list): list of scores, one for each detection
    :return: result (image): image with detection results splashed on it, 3-channel, 8-bit RGB image
    """

    # Opacity of masks: 0.5
    opacity = 0.5

    result = image.astype(float)/255

    for detection_idx in range(masks.shape[2]):

        if not np.any(bboxes[detection_idx]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        # Get the color in float form
        color = colors[class_names[class_ids[detection_idx]]]

        # Draw the segmentation mask
        mask = masks[:,:,detection_idx]
        alpha_mask = np.stack((mask, mask, mask), axis=2)
        alpha_mask = alpha_mask.astype(np.float) * opacity
        assert alpha_mask.shape == image.shape

        foreground = np.ones(image.shape, dtype=float) * color
        _background = cv2.multiply(1.0 - alpha_mask, result)
        _foreground = cv2.multiply(alpha_mask, foreground)

        result = cv2.add(_foreground, _background)

        # Draw the bounding box
        y1, x1, y2, x2 = bboxes[detection_idx]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness=1)

        # Caption time
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        lineType = 2
        offset_x_text = 2
        offset_y_text = -4
        label = class_names[class_ids[detection_idx]]
        caption = "{} {:.3f}".format(label, scores[detection_idx]) if scores.any() else label

        cv2.putText(result, caption, (x1 + offset_x_text, y2 + offset_y_text), fontFace=font, fontScale=fontScale,
                    color=(1.0, 1.0, 1.0), lineType=lineType)

    result *= 255
    result = result.astype(np.uint8)

    return result


def produce_masks(model, config, path, output_path, format = "png"):
    # Check if output path already exists
    try:
        os.makedirs(output_path)
    except OSError:
        print("MaskRCNN: Dir " + output_path + " already exists. Please remove it to repeat the inference process.")
        exit(1)

    # Fix path if require
    if path[-1] != "/":
        path = path + "/"

    # Take all the images paths
    files = glob.glob(path + "*." + format)

    # Process all the images
    for file in files:
        # Compose output name
        file_name = file.split("/")[-1].split(".")[0]

        # Process image
        image = cv2.imread(file)

        # OpenCV returns images as BGR, convert to RGB
        image = image[..., ::-1]

        # Detect objects
        r = model.detect([image], verbose=1)[0]

        # Get image size
        height, width, channels = image.shape

        for cl in dataset.class_info:
            if (any(r['class_ids'] == cl['id'])):
                print("Detected " + cl['name'])
                mask = np.zeros((height, width), dtype = np.uint8)
                masks_ids = np.where(r['class_ids'] == cl['id'])[0]
                # Squash masks from same object
                for id in masks_ids:
                    mask_i = r['masks'][:, :, id]
                    mask_i.astype(np.uint8)
                    mask = np.logical_or(mask, mask_i)

                mask = mask * 255
                try:
                    imageio.imwrite(output_path + "/" + cl['name'] + "_" + file_name + ".png", mask)
                except:
                    print("****************************************************************************")
                    print("SKIPPING: " + cl['name'] + "in frame" + file_name)
                    print("****************************************************************************")
                    pass



def detect_and_splash_results(model, config, dataset, class_colors, image_path=None, image_sequence_path=None, video_path=None):

    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = cv2.imread(image_path)
        # OpenCV returns images as BGR, convert to RGB
        image = image[..., ::-1]
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = apply_detection_results(image, r['masks'], r['rois'], r['class_ids'], dataset.class_names, class_colors, scores=r['scores'])
        # Back to BGR for OPENCV
        splash = splash[..., ::-1]
        # Save output
        file_name = image_path + "_splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        cv2.imwrite(file_name, splash)
    elif video_path:
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = os.path.basename(video_path) + "_splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = apply_detection_results(image, r['masks'], r['rois'], r['class_ids'], dataset.class_names, class_colors, scores=r['scores'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

# TODO: REMOVE THIS IF PYTHON-TKINTER IS INSTALLED ON SERVER
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    import random
    import colorsys

    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

############################################################
#  Main script
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'splash', 'evaluate'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to detect objects on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to detect objects on')
    parser.add_argument('--sequence', required=False,
                        metavar="path to image sequence",
                        help='Sequence of images')
    parser.add_argument('--gpu_id', required=False,
                        metavar="GPU ID",
                        help='Sequence of images')
    parser.add_argument('--number_classes', required=False,
                        metavar="number of classes",
                        help='Sequence of images')
    parser.add_argument('--output', required=False,
                        metavar="",
                        help='Output path')

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    # Instance the proper config file, depending on the dataset to use
    if args.command == "train":
        #config = configurations.TabletopConfigTraining()
        config = configurations.YCBVideoConfigTraining()
    else:
        #config = TabletopConfigInference()
        config = configurations.YCBVideoConfigInference()

    # Override configuration according to parsed arguments
    if args.number_classes is not None:
        config.NUM_CLASSES = int(args.number_classes)

    # Add some env variables to set GPU usage
    if args.gpu_id is not None:
        config.GPU_ID = args.gpu_id
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_ID
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights



    # Load weights
    print("Loading weights ", weights_path)
    if args.command == "train":
        # Exclude the last layers because they require a matching
        # number of classes
        #POSSIBLE BUG: WATCH OUT WHEN RESTARTING A TRAINING SESSION THAT WAS INTERRUPTED!!!!
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, config)
    elif args.command == "splash":

        #TODO: BRING DATASET INSTANTIATION OUT OF THIS IF STATEMENT

        # Automatically discriminate the dataset according to the config file
        if isinstance(config, configurations.TabletopConfigInference):
            # Load the validation dataset
            dataset = datasets.TabletopDataset()
        elif isinstance(config, configurations.YCBVideoConfigInference):
            dataset = datasets.YCBVideoDataset()

        # No need to load the whole dataset, just the class names will be ok
        dataset.load_class_names(args.dataset)

        # Create a dict for assigning colors to each class
        class_colors = {}
        random_class_colors = random_colors(len(dataset.class_names))
        class_colors = {class_id: color for (color, class_id) in zip(random_class_colors, dataset.class_names)}

        detect_and_splash_results(model, image_path=args.image, image_sequence_path=args.image_sequence, video_path=args.video, config=config, dataset=dataset, class_colors=class_colors)

    elif args.command == 'evaluate':
        evaluate_model(model, config)
    elif args.command == 'masks':
        print("Selected command 'masks'")
        dataset = datasets.YCBVideoDataset()
        dataset.load_class_names(args.dataset)
        if args.output is None:
            print('Missing argument output required for command masks.')
            exit (0)
        produce_masks(model, dataset, args.sequence, args.output)
    else:
        print("'{}' is not recognized. "
              "Use 'train', 'splash' or 'evaluate'".format(args.command))
