############################################################
#  Training Configurations
############################################################

from mrcnn.config import Config


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
    LAYERS_TUNE = '4+'


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
    DETECTION_MIN_CONFIDENCE = 0.75


class YCBVideoConfigTraining(Config):
    """Configuration for  training on the YCB_Video dataset for segmentation.
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
    NUM_CLASSES = 1 + 21  # Background + 21 YCB_video objects

    # Specify the backbone network
    BACKBONE = "resnet50"

    # Number of training steps per epoch
    STEPS_PER_EPOCH = None

    # Number of epochs
    #EPOCHS = 50

    # Skip detections with < some confidence level
    DETECTION_MIN_CONFIDENCE = 0.9

    # Define stages to be fine tuned
    LAYERS_TUNE = 'heads'


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
    NUM_CLASSES = 1 + 21  # Background + 21 YCB_video objects

    # Specify the backbone network
    BACKBONE = "resnet50"

    # Skip detections with < some confidence level
    DETECTION_MIN_CONFIDENCE = 0.75

