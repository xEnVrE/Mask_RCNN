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
import json
import datetime
import numpy as np
import skimage.draw
import scipy.io
import cv2
from keras.utils.generic_utils import Progbar

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class TabletopConfigTraining(Config):
    """Configuration for training on the synthetic tabletop dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "synth_tabletop_training"

    # P100s can hold up to 4 images using ResNet50.
    # During inference, make sure to set this to 1.
    IMAGES_PER_GPU = 4

    # Define number of GPUs to use
    GPU_COUNT = 4
    GPU_ID = "0,1,2,3"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 21  # Background + random YCB objects

    # Specify the backbone network
    BACKBONE = "resnet50"

    # Number of training steps per epoch
    STEPS_PER_EPOCH = None

    # Number of epochs
    EPOCHS = 100

    # Skip detections with < some confidence level
    DETECTION_MIN_CONFIDENCE = 0.9

    # Define stages to be fine tuned
    LAYERS_TUNE = '3+'

class TabletopConfigInference(Config):
    """Configuration for training on the synthetic tabletop dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "synth_tabletop_inference"

    # P100s can hold up to 4 images using ResNet50.
    # During inference, make sure to set this to 1.
    IMAGES_PER_GPU = 1

    # Define number of GPUs to use
    GPU_COUNT = 1
    GPU_ID = "0"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 21 # Background + random YCB objects

    # Specify the backbone network
    BACKBONE = "resnet50"

    # Skip detections with < some confidence level
    DETECTION_MIN_CONFIDENCE = 0.9

class YCBVideoConfigTraining(Config):
    """Configuration for training on the YCB_Video dataset for segmentation.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "ycb_video_training"

    # P100s can hold up to 4 images using ResNet50.
    # During inference, make sure to set this to 1.
    IMAGES_PER_GPU = 4

    # Define number of GPUs to use
    GPU_COUNT = 4
    GPU_ID = "0,1,2,3"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 20  # Background + 20 YCB objects (no wood block!)

    # Specify the backbone network
    BACKBONE = "resnet50"

    # Number of training steps per epoch
    STEPS_PER_EPOCH = None

    # Number of epochs
    EPOCHS = 100

    # Skip detections with < some confidence level
    DETECTION_MIN_CONFIDENCE = 0.9

    # Define stages to be fine tuned
    LAYERS_TUNE = '4+'

    REMOVE_WOOD_BLOCK = True

class YCBVideoConfigInference(Config):
    """Configuration for performing inference with the YCB_Video dataset for segmentation.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "ycb_video_inference"

    # P100s can hold up to 4 images using ResNet50.
    # During inference, make sure to set this to 1.
    IMAGES_PER_GPU = 1

    # Define number of GPUs to use
    GPU_COUNT = 1
    GPU_ID = "0"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 20  # Background + 20 YCB objects (no wood block!)

    # Specify the backbone network
    BACKBONE = "resnet50"

    # Skip detections with < some confidence level
    DETECTION_MIN_CONFIDENCE = 0.9

    REMOVE_WOOD_BLOCK = True

############################################################
#  Datasets
############################################################

class YCBVideoDataset(utils.Dataset):

    def parse_class_list(self, dataset_root):
        """
        Parses the class list from the classes.txt file of the dataset.
        Does not include background as a class
        :param dataset_root (string): root directory of the dataset.
        :return: class_list (list): ordered list of classes
        """
        # Classes file is dataset_root/image_sets/classes.txt
        classes_filename = os.path.join(dataset_root, "image_sets", "classes.txt")
        with open(classes_filename) as handle:
            classes_list_raw = handle.readlines()
        classes_list = [cl.strip() for cl in classes_list_raw]

        return classes_list

    def load_class_names(self, dataset_root, verbose=True, remove_wood_block=True):
        """
        Loads the class list into the Dataset class, without opening any metadata file
        :param dataset_root (string): root directory of the dataset.
        """

        # In this dataset, the class id is the 1-based index of the corresponding line in the classes file
        # Background has id 0, but it is not included in the class list
        class_list = self.parse_class_list(dataset_root)

        for class_id, class_name in  enumerate(class_list):
            # exclude wood block!
            if remove_wood_block and (class_name == '036_wood_block'):
                continue
            self.add_class('ycb_video', class_id = class_id+1, class_name = class_name)
            
        self.class_names = [cl['name'] for cl in self.class_info]

        if verbose:
            print("Classes loaded: ", len(self.class_names))
            for cl in self.class_info:
                print("\tID {}:\t{}".format(cl['id'], cl['name']))

    def load_dataset(self, dataset_root, subset, remove_wood_block=True):
        """
        Loads the YCB_Video dataset paths (without opening image files).
        :param dataset_root (string): Root directory of the dataset.
        :param subset (string): Train or validation dataset.
        """

        # Training or validation dataset
        assert subset in ["train", "val"]

        # Add the classes (order is vital for mask id consistency)
        self.load_class_names(dataset_root, remove_wood_block=remove_wood_block)

        # Discriminate between train and validation set
        subset_file = os.path.join(dataset_root, 'image_sets', 'train.txt') if subset == 'train' else os.path.join(dataset_root, 'image_sets', 'val.txt')

        # Iterate over every data element to add images and masks
        with open(subset_file, 'r') as handle:
            frame_file_list_raw = handle.readlines()
        frame_file_list = [fr.strip() for fr in frame_file_list_raw]

        data_dir = os.path.join(dataset_root, 'data/')

        progress_step = max(round(len(frame_file_list)/1000), 1)
        progbar = Progbar(target = len(frame_file_list))
        print("Loading ", subset, "dataset...")

        for progress_idx, frame in enumerate(frame_file_list):
            rgb_image_path = data_dir + frame + '-color.png'
            mask_path = data_dir + frame + '-label.png'
            metadata_path = data_dir + frame + '-meta.mat'

            # Load the metadata file for each sample to get the instance IDs for the masks
            metadata = scipy.io.loadmat(metadata_path)
            instance_ids =  metadata['cls_indexes']
            instance_ids = instance_ids.reshape(instance_ids.size)

            if remove_wood_block:
                instance_ids[instance_ids != 16]
                shift_ids = -1 * (instance_ids > 16)
                instance_ids = instance_ids.astype(np.int64) + shift_ids
                instance_ids = instance_ids.astype(np.uint8)

            # Add an image to the dataset
            if os.path.isfile(rgb_image_path) and os.path.isfile(mask_path):
                self.add_image(
                    "ycb_video",
                    image_id = frame,
                    path = rgb_image_path,
                    width = 640, height = 480,
                    mask_path = mask_path,
                    mask_ids = instance_ids
                )

            # Keep track of progress
            if progress_idx%progress_step == 0 or progress_idx == len(frame_file_list):
                progbar.update(progress_idx+1)

        print("\nDataset loaded: ", len(self.image_info), "images found.")

    def load_mask(self, image_id):
        """Generate instance mask array for an image id
        :param
            image_id (string): id of the image, according to self.image_info list
        :return:
            masks (ndarray): A bool array of shape [height, width, instance count] with
                    one mask per instance.
            class_ids (ndarray): A 1D array of class IDs of the instance masks.
        """

        # If not a YCB_Video dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "ycb_video":
            return super(self.__class__, self).load_mask(image_id)

        # Instance ids have already been loaded
        # This dataset is easier because there can only be one instance of each object.
        # Therefore, grayscale mask ids are directly related to the class
        class_ids = image_info['mask_ids']

        # Load image mask
        mask_image = skimage.io.imread(image_info["mask_path"])

        # Create empty tensor
        no_of_masks = class_ids.size
        assert no_of_masks > 0

        masks = np.zeros((image_info["height"], image_info["width"], no_of_masks),
                            dtype=np.bool)

        # create boolean masks for each instance, respecting instance id order
        for idx in range(no_of_masks):
            masks[:, :, idx] = mask_image == class_ids[idx]

        return masks, class_ids

    def get_class_id(self, image_text_label):
        """Return class id according to the image textual label
        Returns:
            class_id: int of the class id according to self.class_info. -1
                if class not found
        """
        for this_class in self.class_info:
            if this_class["name"] != image_text_label:
                continue
            else:
                return this_class["id"]

        return -1

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "ycb_video":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

class TabletopDataset(utils.Dataset):

    def parse_class_list(self, dataset_root):
        """
        Parses the class list from the meta file of the dataset.
        Does not include background as a class
        :param dataset_root (string): root directory of the dataset
        :return: class_list (list): ordered list of classes
        """
        # Very inefficient, needs to open the whole json file!
        # Load classes from the json file
        for subset in ['test', 'val', 'train']:
            DATASET_JSON_FILENAME = os.path.join(dataset_root, subset, 'dataset.json')
            if os.path.isfile(DATASET_JSON_FILENAME):
                break
        # Assertion error if there is no json file for the dataset
        assert os.path.isfile(DATASET_JSON_FILENAME)

        # Open the metadata file
        with (open(DATASET_JSON_FILENAME, 'r')) as handle:
            dataset_dict = json.loads(json.load(handle))

        # Add classes (except __background__, that is added by default)
        # We need to make sure that the classes are added according to the order of their IDs in the dataset
        # Or the names will be screwed up
        class_entries_sorted_by_id = sorted(dataset_dict['Classes'].items(), key=lambda kv: kv[1])

        class_list = [cls[0] for cls in class_entries_sorted_by_id]

        return class_list

    def load_class_names(self, dataset_root, verbose=True):
        """
        Loads the class list into the Dataset class, without opening any metadata file
        :param class_list (list): list of class names (order defines the id)
        """

        # In this dataset, the class id is the 1-based index of the corresponding line in the classes file
        # Background has id 0, but it is not included in the class list
        class_list = self.parse_class_list(dataset_root)

        for class_id, class_name in enumerate(class_list):
            if class_name == '__background__':
                continue
            self.add_class('tabletop', class_id = class_id, class_name = class_name)

        self.class_names = [cl['name'] for cl in self.class_info]
        if verbose:
            print("Classes loaded: ", len(self.class_names))
            for cl in self.class_info:
                print("\tID {}:\t{}".format(cl['id'], cl['name']))

    def load_dataset(self, dataset_root, subset):
        """
        Load the tabletop dataset.
        :param dataset_root (string): Root directory of the dataset.
        :param subset (string): Train or validation dataset
        """

        # Training or validation
        assert subset in ["train", "val"]
        subset_dir = os.path.join(dataset_root, subset)

        # Load dataset metadata
        DATASET_JSON_FILENAME = os.path.join(subset_dir, "dataset.json")
        assert os.path.isfile(DATASET_JSON_FILENAME)

        with (open(DATASET_JSON_FILENAME, 'r')) as handle:
            dataset_dict = json.loads(json.load(handle))

        self.load_class_names(dataset_root)

        # fix the maskID field
        for path, info in dataset_dict['Images'].items():
            fixed_mask_id = {}
            for key, value in info['MaskID'].items():
                fixed_mask_id[int(key)] = value
            dataset_dict['Images'][path]['MaskID'] = fixed_mask_id

        # The dataset dictionary is organized as follows:
        # {
        #   "Classes": {
        #       "__background__" : 0
        #       "class_name" : 1
        #       ...
        #   }
        #   "Images": {
        #       "image_1_filename": {
        #           "Annotations":"path_to_annotation_1.xml"
        #           "MaskPath":"path_to_mask_1.png"
        #           "MaskID":{
        #               id_0:"class_name"
        #               ...
        #           }
        #       ...
        #   }
        #
        # Annotations = bounding boxes of object instances in the image
        # MaskID = correspondences between mask colors and class label

        progress_step = max(round(len(dataset_dict['Images'].keys()) / 1000), 1)
        progbar = Progbar(target=len(dataset_dict['Images'].keys()))

        print("Loading ", subset, "dataset...")

        # Iterate over images in the dataset to add them
        for progress_idx, (path, info) in enumerate(dataset_dict['Images'].items()):
            image_path = os.path.join(subset_dir, path)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "tabletop",
                image_id = image_path,
                path = image_path,
                width = width, height = height,
                mask_path = os.path.join(subset_dir, info['MaskPath']),
                mask_ids = info['MaskID'])

            # Keep track of progress
            if progress_idx%progress_step == 0 or progress_idx == len(dataset_dict['Images'].keys()):
                progbar.update(progress_idx+1)

    def get_class_id(self, image_text_label):
        """Return class id according to the image textual label
        Returns:
            class_id: int of the class id according to self.class_info. -1
                if class not found
        """
        for this_class in self.class_info:
            if this_class["name"] != image_text_label:
                continue
            else:
                return this_class["id"]

        return -1

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                    one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a tabletop dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "tabletop":
            return super(self.__class__, self).load_mask(image_id)

        mask_image = skimage.io.imread(image_info["mask_path"])
        mask_classes = image_info["mask_ids"]

        # Create empty return values
        masks = np.zeros( (image_info["height"], image_info["width"], len(mask_classes.keys())),
                            dtype=np.bool)
        class_ids = np.zeros(len(mask_classes.keys()), dtype=np.int32)

        # The dataset already contains binary maps, we just need to extract
        # them from the .png mask according to their ID
        # The ID in the mask file is different for each instance, therefore
        # we need to refer to the text label and find out the ID in
        # self.class_info
        current_inst = 0
        for instance_id, instance_class_label in mask_classes.items():
            this_instance_id = self.get_class_id(instance_class_label)
            # enforce ids to be positive!
            assert this_instance_id > 0
            masks[:, :, current_inst] = mask_image == instance_id
            class_ids[current_inst] = this_instance_id
            current_inst += 1

        return masks, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "tabletop":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Utils
############################################################

def train(model, config):
    """Train the model."""

    # Automatically discriminate the dataset according to the config file
    if isinstance(config, TabletopConfigTraining):
        # Load the training dataset
        dataset_train = TabletopDataset()
        # Load the validation dataset
        dataset_val = TabletopDataset()
    elif isinstance(config, YCBVideoConfigTraining):
        dataset_train = YCBVideoDataset()
        dataset_val = YCBVideoDataset()

    dataset_train.load_dataset(args.dataset, "train")
    dataset_train.prepare()

    dataset_val.load_dataset(args.dataset, "val")
    dataset_val.prepare()

    # Experimental: train/validate on whole dataset.
    # Number of steps must be equal to round_down(dataset_size/batch_size)
    if config.STEPS_PER_EPOCH == None:
        config.STEPS_PER_EPOCH = int(dataset_train.num_images/config.BATCH_SIZE)

    # TRAINING SCHEDULE
    stages_trained = '3+'
    print("Training network stages" + stages_trained)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers=stages_trained)
    stages_trained = '4+'
    print("Training network stages" + stages_trained)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/5.0,
                epochs=20,
                layers=stages_trained)
    stages_trained = 'heads'
    print("Training network stages" + stages_trained)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/10.0,
                epochs=10,
                layers=stages_trained)


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
        file_name = os.path.basename + "_splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
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
    if isinstance(config, TabletopConfigInference):
        # Load the validation dataset
        dataset_val = TabletopDataset()
    elif isinstance(config, YCBVideoConfigInference):
        dataset_val = YCBVideoDataset()

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
        #config = TabletopConfigTraining()
        config = YCBVideoConfigTraining()
    else:
        #config = TabletopConfigInference()
        config = YCBVideoConfigInference()

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
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, config)
    elif args.command == "splash":

        #TODO: BRING DATASET INSTANTIATION OUT OF THIS IF STATEMENT

        # Automatically discriminate the dataset according to the config file
        if isinstance(config, TabletopConfigInference):
            # Load the validation dataset
            dataset = TabletopDataset()
        elif isinstance(config, YCBVideoConfigInference):
            dataset = YCBVideoDataset()

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
