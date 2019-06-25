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
"""

import os
import sys
import datetime
import numpy as np
import cv2
from keras.utils.generic_utils import Progbar
import imgaug

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import config and dataset files
from samples.humanoids_pouring import configurations
from samples.humanoids_pouring import datasets

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

def train(model, config):
    """Train the model."""

    # Automatically discriminate the dataset according to the config file
    if isinstance(config, configurations.TabletopConfigTraining):
        # Load the training dataset
        dataset_train = datasets.TabletopDataset()
        # Load the validation dataset
        dataset_val = datasets.TabletopDataset()
    elif isinstance(config, configurations.YCBVideoConfigTraining):
        dataset_train = datasets.YCBVideoDataset()
        dataset_val = datasets.YCBVideoDataset()

    dataset_train.load_dataset(args.dataset, "train")
    dataset_train.prepare()

    dataset_val.load_dataset(args.dataset, "val")
    dataset_val.prepare()

    # Experimental: train/validate on whole dataset.
    # Number of steps must be equal to round_down(dataset_size/batch_size)
    if config.STEPS_PER_EPOCH == None:
        config.STEPS_PER_EPOCH = int(dataset_train.num_images/config.BATCH_SIZE)

    augmentation = imgaug.augmenters.Sequential([
        imgaug.augmenters.Fliplr(0.5),                          # Horizontal flips
        imgaug.augmenters.Sometimes(0.5,
            imgaug.augmenters.GaussianBlur(sigma=(0, 0.5))      # Gaussian blur
        ),
        imgaug.augmenters.Affine(
            rotate=(-90,90),                                    # Apply rotation
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}            # Scale change
        ),
        imgaug.augmenters.ContrastNormalization((0.8, 1.2))    # Change image contrast
    ])

    # TRAINING SCHEDULE

    stages_trained = 'heads'
    print("Training network stages" + stages_trained)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers=stages_trained,
                augmentation=augmentation)

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

def detect_and_splash_results(model, config, dataset, class_colors, image_path=None, video_path=None):

    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = cv2.imread(args.image)
        # OpenCV returns images as BGR, convert to RGB
        image = image[..., ::-1]
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = apply_detection_results(image, r['masks'], r['rois'], r['class_ids'], dataset.class_names, class_colors, scores=r['scores'])
        # Back to BGR for OPENCV
        splash = splash[..., ::-1]
        # Save output
        file_name = os.path.basename(image_path) + "_splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
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

def evaluate_model(model, config):
    """
    Evaluate the loaded model on the target dataset
    :param model: the architecture model, with loaded weights
    :param config: the configuration class for this dataset
    """

    # Automatically discriminate the dataset according to the config file
    if isinstance(config, configurations.TabletopConfigInference):
        # Load the validation dataset
        dataset_val = datasets.TabletopDataset()
    elif isinstance(config, configurations.YCBVideoConfigInference):
        dataset_val = datasets.YCBVideoDataset()

    dataset_val.load_dataset(args.dataset, "val")
    dataset_val.prepare()

    # Compute COCO-Style mAP @ IoU=0.5-0.95 in 0.05 increments
    # Running on all images
    #image_ids = np.random.choice(dataset_val.image_ids, 200)
    image_ids = dataset_val.image_ids
    APs = []
    AP50s = []
    AP75s = []
    image_batch_vector = []
    image_batch_eval_data = []
    img_batch_count = 0

    import time
    t_inference = 0

    class image_eval_data:
        def __init__(self, image_id, gt_class_id, gt_bbox, gt_mask):
            self.IMAGE_ID = image_id
            self.GT_CLASS_ID = gt_class_id
            self.GT_BBOX = gt_bbox
            self.GT_MASK = gt_mask
            self.DETECTION_RESULTS = None

    print("Evaluating model...")
    progbar = Progbar(target = len(image_ids))


    for idx, image_id in enumerate(image_ids):
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, config,
                                   image_id, use_mini_mask=False)

        # Compose a vector of images and data
        image_batch_vector.append(image)
        image_batch_eval_data.append(image_eval_data(image_id, gt_class_id, gt_bbox, gt_mask))
        img_batch_count += 1

        # If a batch is ready, go on to detection
        # The last few images in the dataset will not be used if batch size > 1
        if img_batch_count < config.BATCH_SIZE:
            continue

        # Run object detection
        t_start = time.time()
        results = model.detect(image_batch_vector, verbose=0)
        t_inference += (time.time() - t_start)

        assert len(image_batch_eval_data) == len(results)

        for eval_data, detection_results in zip(image_batch_eval_data, results):
            eval_data.DETECTION_RESULTS = detection_results
            # Compute mAP at different IoU (as msCOCO mAP is computed)
            AP = utils.compute_ap_range(eval_data.GT_BBOX, eval_data.GT_CLASS_ID, eval_data.GT_MASK,
                                        eval_data.DETECTION_RESULTS["rois"],
                                        eval_data.DETECTION_RESULTS["class_ids"],
                                        eval_data.DETECTION_RESULTS["scores"],
                                        eval_data.DETECTION_RESULTS['masks'],
                                        verbose=0)
            AP50, _, _, _ = utils.compute_ap(eval_data.GT_BBOX, eval_data.GT_CLASS_ID, eval_data.GT_MASK,
                                        eval_data.DETECTION_RESULTS["rois"],
                                        eval_data.DETECTION_RESULTS["class_ids"],
                                        eval_data.DETECTION_RESULTS["scores"],
                                        eval_data.DETECTION_RESULTS['masks'],
                                        iou_threshold=0.5)
            AP75, _, _, _ = utils.compute_ap(eval_data.GT_BBOX, eval_data.GT_CLASS_ID, eval_data.GT_MASK,
                                        eval_data.DETECTION_RESULTS["rois"],
                                        eval_data.DETECTION_RESULTS["class_ids"],
                                        eval_data.DETECTION_RESULTS["scores"],
                                        eval_data.DETECTION_RESULTS['masks'],
                                        iou_threshold=0.75)

            APs.append(AP)
            AP50s.append(AP50)
            AP75s.append(AP75)

        # Reset the batch info
        image_batch_vector = []
        image_batch_eval_data = []
        img_batch_count = 0

        progbar.update(idx+1)

    print("\nmAP[0.5::0.05::0.95]: ", np.mean(APs))
    print("mAP[0.5]: ", np.mean(AP50s))
    print("mAP[0.75]: ", np.mean(AP75s))

    print("Inference time for", len(image_ids), "images: ", t_inference, "s \tAverage FPS: ", len(image_ids)/t_inference)

    return APs

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

    # Add some env variables to set GPU usage
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
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

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

        detect_and_splash_results(model, image_path=args.image,
                                video_path=args.video, config=config, dataset=dataset, class_colors=class_colors)

    elif args.command == 'evaluate':
        evaluate_model(model, config)
    else:
        print("'{}' is not recognized. "
              "Use 'train', 'splash' or 'evaluate'".format(args.command))
