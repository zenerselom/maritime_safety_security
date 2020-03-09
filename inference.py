'''
Contains code for working with the Inference Engine.
You'll learn how to implement this code and more in
the related lesson on the topic.
'''

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

class Network:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None


    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(network=self.network,num_requests=2,device_name=device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        return
        
    def check_device_extension(self,log,device="CPU"):
        if "CPU" in device:
            supported_layers = self.plugin.query_network(self.network, "CPU")
            not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
               log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
               log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
               sys.exit(1)

        assert len(self.network.inputs.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"
        return
        
    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs[self.input_blob].shape


    def async_inference(self, image,request_id=0):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        ### TODO: Start asynchronous inference
        
        self.infer_request = self.exec_network.start_async(request_id = request_id,inputs={self.input_blob:image})
        
        return


    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        ### TODO: Wait for the async request to be complete
       
        status = self.exec_network.requests[0].wait(-1)
        
        return status


    def extract_output(self,request_id=0):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        ### TODO: Return the outputs of the network from the output_blob
        return self.exec_network.requests[request_id].outputs
