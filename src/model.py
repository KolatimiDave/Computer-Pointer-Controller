import os
import logging as log
from openvino.inference_engine import IENetwork, IECore

class Model():
    '''
    Class for the Gaze estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, prob_thresh = 0.5, visual_flag = ''):
        '''
        Use this to set your instance variables.
        '''
        self.plugin = None
        self.network = None
        self.device = device
        self.extensions = extensions
        self.exec_net = None
        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.output_shape = None
        self.model_xml = model_name
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        self.prob_thresh = prob_thresh
        self.visual_flag = visual_flag
    
        
    def check_model(self):
        '''
        Reads in the IR format of the model
        '''
        try:
            self.network = self.plugin.read_network(self.model_xml,self.model_bin)
        except Exception:
            raise ValueError('Error on Reading the IR, Ensure correct path to IR is given')
            

    def check_layers(self):
        '''
        Check for supported and unsupported layers in our network 
        '''
        try:
            supported_layers = self.plugin.query_network(network = self.network, device_name = self.device)
            unsupported_layers = [ i for i in self.network.layers.keys() if i not in supported_layers]
            if len(unsupported_layers)!=0 and self.device == 'CPU':
                self.plugin.add_extentions(self.extensions, self.device)
        except Exception:
            log.error('Unable to load the following unsupported layers')
            exit()


    def load_model(self):
        # Initialize the plugin 
        self.plugin = IECore()

        # load the IR
        self.check_model()

        # check supported and unsupported layers
        self.check_layers()

        # Load the network into the plugin
        self.exec_net = self.plugin.load_network(network = self.network, device_name = self.device, num_requests = 1)

        # Get the input layer 
        self.input_blob = next(iter(self.network.inputs))

        # Get the input shape
        self.input_shape = self.network.inputs[self.input_blob].shape

        # Get the output layer
        self.output_blob = next(iter(self.network.outputs))

        # Get the output shape
        self.output_shape = self.network.outputs[self.output_blob].shape