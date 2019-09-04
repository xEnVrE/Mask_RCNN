############################################################
#  Datasets
############################################################
import os
import sys
import json, pickle
import numpy as np
import skimage.draw
import scipy.io
from keras.utils.generic_utils import Progbar

# To find local version of the library
from mrcnn import model as modellib, utils

class YCBVideoDataset(utils.Dataset):

    # List of classes to remove
    UNWANTED_CLASS_LIST = {}

    def get_dataset_logfile(self, dataset_root, class_number, num_images, subset):
        """
        Returns the path of the logfile that should be created for this dataset.
        :param dataset_root (string): root directory of the dataset
        :param class_number (int): number of classes in the dataset, including bg
        :param num_images (int): number of images in the dataset
        :param subset (string): typically train or val
        :return dataset_logfile (string): path of the logfile
        """

        # Compose the name of the logfile
        logfile_name = os.path.join(dataset_root,
                                    "image_sets",
                                    "dataset_logfiles",
                                    "YCBVideo_"+subset+"_"+str(class_number)+"_"+str(num_images)+"_log")

        return logfile_name

    def dump_to_log(self, logfile_name):
        """
        Dumps the dataset class.
        :param logfile_name (string): path of the logfile
        """

        log_dict = {"_image_ids" : self.image_ids,
                    "image_info" : self.image_info,
                    "class_info" : self.class_info,
                    "source_class_ids" : self.source_class_ids}

        logfile_dirname = os.path.dirname(logfile_name)        

        if not os.path.exists(logfile_dirname):
            os.makedirs(logfile_dirname)

        with open(logfile_name, 'wb') as handle:
            pickle.dump(log_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return


    def load_from_log(self, logfile_name):
        """
        Loads the dataset class.
        :param logfile_name (string): path of the logfile
        """

        with open(logfile_name, 'rb') as handle:
            log_dict = pickle.load(handle)

        self._image_ids = log_dict["_image_ids"]
        self.image_info = log_dict["image_info"]
        self.class_info = log_dict["class_info"]
        self.source_class_ids = log_dict["source_class_ids"]

        return

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

        # Remove unwanted classes from class list
        for unwanted_class in self.UNWANTED_CLASS_LIST.keys():
            if self.UNWANTED_CLASS_LIST and (unwanted_class in classes_list):
                classes_list.remove(unwanted_class)

        return classes_list

    def load_class_names(self, dataset_root, verbose=True):
        """
        Loads the class list into the Dataset class, without opening any metadata file
        :param dataset_root (string): root directory of the dataset.
        """

        # In this dataset, the class id is the 1-based index of the corresponding line in the classes file
        # Background has id 0, but it is not included in the class list
        class_list = self.parse_class_list(dataset_root)

        for class_id, class_name in enumerate(class_list):
            self.add_class('ycb_video', class_id=class_id+1, class_name=class_name)

        self.class_names = [cl['name'] for cl in self.class_info]

        if verbose:
            print("Classes loaded: ", len(self.class_names))
            for cl in self.class_info:
                print("\tID {}:\t{}".format(cl['id'], cl['name']))

    def load_dataset(self, dataset_root, subset):
        """
        Loads the YCB_Video dataset paths (without opening image files).
        :param dataset_root (string): Root directory of the dataset.
        :param subset (string): Train or validation dataset.
        """

        # Training or validation dataset
        assert subset in ["train", "val"]

        # Add the classes (order is vital for mask id consistency)
        self.load_class_names(dataset_root)

        # Discriminate between train and validation set
        subset_file = os.path.join(dataset_root, 'image_sets', 'train.txt') if subset == 'train' else os.path.join(dataset_root, 'image_sets', 'val.txt')

        # Iterate over every data element to add images and masks
        with open(subset_file, 'r') as handle:
            frame_file_list_raw = handle.readlines()
        frame_file_list = [fr.strip() for fr in frame_file_list_raw]

        data_dir = os.path.join(dataset_root, 'data/')

        # Check if a logfile exists for the dataset
        dataset_logfile = self.get_dataset_logfile(dataset_root, len(self.class_names), len(frame_file_list), subset)
        print("Looking for logfile: ", dataset_logfile)
        dataset_logfile_found = os.path.isfile(dataset_logfile)

        if not dataset_logfile_found:

            # Load all image data
            progress_step = max(round(len(frame_file_list)/1000), 1)
            progbar = Progbar(target = len(frame_file_list))
            print("Loading ", subset, "dataset...")

            for progress_idx, frame in enumerate(frame_file_list):
                rgb_image_path = data_dir + frame + '-color.png'
                mask_path = data_dir + frame + '-label.png'
                metadata_path = data_dir + frame + '-meta.mat'

                # Load the metadata file for each sample to get the instance IDs for the masks
                metadata = scipy.io.loadmat(metadata_path)
                instance_ids = metadata['cls_indexes']
                instance_ids = instance_ids.reshape(instance_ids.size)

                # Remove detection ids related to unwanted classes from detection list
                if self.UNWANTED_CLASS_LIST:
                    for unwanted_class_name, unwanted_class_id in self.UNWANTED_CLASS_LIST.items():
                        instance_ids = instance_ids[instance_ids != unwanted_class_id]
                        instance_ids = instance_ids - 1 * (instance_ids > unwanted_class_id)

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

            self.dump_to_log(dataset_logfile)

        else:
            print("Pickled dataset found!")

            # THIS WILL PROBABLY NOT WORK!
            self.load_from_log(dataset_logfile)

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

        # Change mask grayscales according to undesired classes
        # TODO: fix this for a greater number of unwanted classes!
        if self.UNWANTED_CLASS_LIST:
            for unwanted_class, unwanted_id in self.UNWANTED_CLASS_LIST.items():
                if np.any(class_ids >= unwanted_id):
                    # Erase any ground truth for unwanted classes
                    mask_image[mask_image == unwanted_id] = 0
                    # Take 1 away from every ground truth with id > unwanted_id
                    mask_fixes = -1*(mask_image>unwanted_id)
                    mask_image = mask_image.astype(np.int) + mask_fixes
                    mask_image = mask_image.astype(np.uint)

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
