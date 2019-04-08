"""
YARP module to test the trained Mask R-CNN model online with live camera data.

Refer to the README.md of the repo for the setup of the requirements of the model. Also, refer to the YARP documentation
 in order to install YARP Python bindings (http://www.yarp.it/yarp_swig.html)

Author: Fabrizio Bottarel (fabrizio.bottarel@iit.it)
"""

import os
import sys
import argparse
import numpy as np

#   Root directory of the project
ROOT_DIR = os.path.abspath("../../")

#   Import Mask R-CNN
sys.path.append(ROOT_DIR)
import mrcnn.model as modellib

#   Import the tabletop dataset custom configuration
import tabletop

#   Declare directories for weights and logs
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

#   Import YARP bindings
YARP_BUILD_DIR = "/home/fbottarel/robot-code/yarp/build"
YARP_BINDINGS_DIR = os.path.join(YARP_BUILD_DIR, "lib/python")

if YARP_BINDINGS_DIR not in sys.path:
    sys.path.insert(0, YARP_BINDINGS_DIR)

import yarp
yarp.Network.init()

#   Add environment variables depending on the system
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#   Set an upper bound to the GPU memory we can use
import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)
K.set_session(sess)

class MaskRCNNWrapperModule (yarp.RFModule):

    def __init__(self, args):
        '''
        Initialize the module with None everywhere.
        The configure() method will be used to set everything up
        '''

        yarp.RFModule.__init__(self)

        self.__rf = None

        self._input_buf_image = None
        self._input_buf_array = None

        self._output_buf_image = None
        self._output_buf_array = None

        self._port_out_bboxes = None
        self._port_out = None
        self._port_in = None
        self._port_rpc = None

        self._module_name = args.module_name

        self._input_img_width = args.input_img_width
        self._input_img_height = args.input_img_height

        self._model_weights_path = os.path.join(MODEL_DIR, args.model_weights_path)

        self._model = None

        self._dataset = None

        self._class_colors = None

    def configure (self, rf):
        '''
        Configure the module internal variables and ports according to resource finder
        '''

        self._rf = rf

        #   Input
        #   Image port initialization
        self._port_in = yarp.BufferedPortImageRgb()
        self._port_in.open('/' +  self._module_name + '/RGBimage:i')

        #   Input buffer initialization
        self._input_buf_image = yarp.ImageRgb()
        self._input_buf_image.resize(self._input_img_width, self._input_img_height)
        self._input_buf_array = np.ones((self._input_img_height, self._input_img_width, 3), dtype = np.uint8)
        self._input_buf_image.setExternal(self._input_buf_array,
                                          self._input_buf_array.shape[1],
                                          self._input_buf_array.shape[0])

        print('Input image buffer configured')

        #   Output
        #   Output image port initialization
        self._port_out = yarp.Port()
        self._port_out.open('/' + self._module_name + '/RGBImage:o')

        #   Output blobs port initialization
        self._port_out_bboxes = yarp.Port()
        self._port_out_bboxes.open('/' + self._module_name + '/bboxes:o')

        #   Output buffer initialization
        self._output_buf_image = yarp.ImageRgb()
        self._output_buf_image.resize(self._input_img_width, self._input_img_height)
        self._output_buf_array = np.zeros((self._input_img_height, self._input_img_width, 3), dtype = np.uint8)
        self._output_buf_image.setExternal(self._output_buf_array,
                                           self._output_buf_array.shape[1],
                                           self._output_buf_array.shape[0])

        print('Output image buffer configured')

        #   RPC port initialization
        self._port_rpc = yarp.RpcServer()
        self._port_rpc.open('/' + self._module_name + '/rpc')
        self.attach_rpc_server(self._port_rpc)

        #   Inference model setup
        #   Configure some parameters for inference
        config = tabletop.YCBVideoConfigInference()
        config.POST_NMS_ROIS_INFERENCE        =300
        config.PRE_NMS_LIMIT                  =1000
        config.DETECTION_MAX_INSTANCES        =10
        config.DETECTION_MIN_CONFIDENCE       =0.8
        
        config.display()

        self._model = modellib.MaskRCNN(mode='inference',
                                  model_dir=MODEL_DIR,
                                  config=config)

        self._detection_results = None

        print('Inference model configured')

        #   Load class names
        dataset_root = os.path.join(ROOT_DIR, "datasets", "YCB_Video_Dataset")

        # Automatically discriminate the dataset according to the config file
        if isinstance(config, tabletop.TabletopConfigInference):
            # Load the validation dataset
            self._dataset = tabletop.TabletopDataset()
        elif isinstance(config, tabletop.YCBVideoConfigInference):
            self._dataset = tabletop.YCBVideoDataset()

        # No need to load the whole dataset, just the class names will be ok
        self._dataset.load_class_names(dataset_root)

        # Create a dict for assigning colors to each class
        random_class_colors = tabletop.random_colors(len(self._dataset.class_names))
        self._class_colors = {class_id: color for (color, class_id) in zip(random_class_colors, self._dataset.class_names)}

        #   Load model weights
        try:
            assert os.path.exists(self._model_weights_path)
        except AssertionError as error:
            print("Model weights path invalid: file does not exist")
            print(error)
            return False

        self._model.load_weights(self._model_weights_path, by_name=True)

        print("Model weights loaded")

        return True

    def interruptModule(self):

        self._port_in.interrupt()
        self._port_out.interrupt()
        self._port_out_bboxes.interrupt()

        return True

    def close(self):

        self._port_in.close()
        self._port_out.close()
        self._port_out_bboxes.close()

        return True

    def getPeriod(self):

        return 0.0

    def updateModule(self):
        '''
        During module update, acquire a streamed image, perform inference using the model and then
        return/display results
        '''

        input_img = self._port_in.read()
        if input_img is None:
            print('Invalid input image (image is None)')
        else:
            self._input_buf_image.copy(input_img)
            assert self._input_buf_array.__array_interface__['data'][0] == self._input_buf_image.getRawImage().__int__()

            #   run detection/segmentation on frame
            frame = self._input_buf_array
            results = self._model.detect([frame], verbose=0)

            # Visualize and stream results
            r = results[0]
            self._detection_results = r
            if len(r['rois']) > 0:
                frame_with_detections = tabletop.apply_detection_results(frame, r['masks'], r['rois'], r['class_ids'],
                                                                         self._dataset.class_names,
                                                                         self._class_colors,
                                                                         scores=r['scores'])

                b = yarp.Bottle()
                for detection_bbox in r['rois']:
                    y1, x1, y2, x2 = detection_bbox
                    bb = b.addList()
                    bb.addDouble(float(x1))
                    bb.addDouble(float(y1))
                    bb.addDouble(float(x2))
                    bb.addDouble(float(y2))
                    
                self._output_buf_array = frame_with_detections.astype(np.uint8)
                self._port_out.write(self._output_buf_image)
                self._port_out_bboxes.write(b)
            else:
                # If nothing is detected, just pass the video frame through
                self._output_buf_array = frame.astype(np.uint8)
                self._port_out.write(self._output_buf_image)

        return True

    def get_component_around(self, seed_x, seed_y):
        '''
        Return a list of points belonging to a detected object, starting from a seed point
        :param seed_x (int): seed point x coordinate
        :param seed_y (int): seed point y coordinate
        :return (list): list of [x,y] points pertaining to the segmented object. Empty if seed is outside the component
        '''

        blob_point_list = []

        #   Assert seed point is within image boundaries
        if not ((seed_x > 0 and seed_x < self._input_img_width) and (seed_y > 0 and seed_y < self._input_img_height)):
            return blob_point_list
        
        #   Assert if seed point is contained in any detection mask
        detection_masks = self._detection_results['masks']
        if not np.any(detection_masks[seed_y, seed_x, :]):
            return blob_point_list

        #   Given seed point is contained in a mask, retrieve such mask
        #   If the seed point is contained in more than one mask, return the first found mask
        #TODO: THIS IS BRUTAL, MAYBE REWRITE THIS SO YOU DON'T LOOK LIKE A CAVEMAN
        for mask_idx in range(detection_masks.shape[2]):
            if detection_masks[seed_y, seed_x, mask_idx]:
                point_array_row, point_array_col = np.where(detection_masks[:,:,mask_idx])
                for point_idx in range(point_array_row.shape[0]):
                    #   The points are enlisted as [x, y] coordinates so row and column order is swapped
                    blob_point_list.append([point_array_col[point_idx], point_array_row[point_idx]])
                break

        #   If none is found, return empty list of points
        return blob_point_list

    def respond(self, command, reply):
        '''
        Respond to rpc commands
        :param command (yarp.Bottle): bottle containing the command (as string)
        :param reply (yarp.Bottle): bottle containing the response (format depending on command)
        '''

        #   Declare available commands
        available_commands = ['get_component_around']

        command_string = command.get(0).toString()
        reply.clear()

        pointlist = reply.addList()

        if command_string not in available_commands:
            print('Command not recognized!')
            return True

        if command_string == available_commands[0]:
            #   return binary object around seed pixel
            seed_x = command.get(1).asInt()
            seed_y = command.get(2).asInt()

            point_list = self.get_component_around(seed_x, seed_y)

            #   Iterate over all points and add them to the bottle as lists
            if point_list:
                for point in point_list:
                    p = pointlist.addList()
                    p.addInt(int(point[0]))
                    p.addInt(int(point[1]))

        return True


def parse_args():
    '''
    Parser for command line input arguments
    :return: input arguments
    '''

    parser = argparse.ArgumentParser(description='Mask R-CNN live demo')

    parser.add_argument('--name', dest='module_name', help='YARP module name',
                        default='instanceSegmenter', type=str)
    parser.add_argument('--width', dest='input_img_width', help='Input image width',
                        default=640, type=int)
    parser.add_argument('--height', dest='input_img_height', help='Input image height',
                        default=480, type=int)
    parser.add_argument(dest='model_weights_path', help='Model weights path relative to the directory PROJECT_ROOT/logs',
			type=str)

    return parser.parse_args()

if __name__ == '__main__':

    #   Parse arguments
    args = parse_args()

    yarp.Network.init()

    detector = MaskRCNNWrapperModule(args)

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefault('name', 'instanceSegmenter')

    rf.configure(sys.argv)

    print('Configuration complete')
    detector.runModule(rf)
