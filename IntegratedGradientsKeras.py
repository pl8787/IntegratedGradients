################################################################
# Implemented by Naozumi Hiranuma (hiranumn@uw.edu)            #
#                                                              #
# Keras-compatible implmentation of Integrated Gradients       # 
# proposed in "Axiomatic attribution for deep neuron networks" #
# (https://arxiv.org/abs/1703.01365).                          #
#                                                              #
# Keywords: Shapley values, interpretable machine learning     #
################################################################

from __future__ import division, print_function
import numpy as np
from time import sleep
import sys
import keras.backend as K

from keras.models import Model, Sequential

'''
Integrated gradients approximates Shapley values by integrating partial
gradients with respect to input features from reference input to the
actual input. The following class implements the paper "Axiomatic attribution
for deep neuron networks".
'''
class integrated_gradients:
    # model: Keras model that you wish to explain.
    # outchannels: In case the model are multi tasking, you can specify which output you want explain .
    def __init__(self, model, outchannels=[], verbose=1, vis_nodes=None, extra_nodes=None):
    
        #get backend info (either tensorflow or theano)
        self.backend = K.backend()
        
        #load model supports keras.Model and keras.Sequential
        if isinstance(model, Sequential):
            self.model = model.model
        elif isinstance(model, Model):
            self.model = model
        else:
            print("Invalid input model")
            return -1

        self.vis_nodes = vis_nodes
        if not self.vis_nodes:
            self.vis_nodes = self.model.inputs        

        #load input tensors
        self.input_tensors = []
        for i in self.model.inputs:
            self.input_tensors.append(i)
        # The learning phase flag is a bool tensor (0 = test, 1 = train)
        # to be passed as input to any Keras function that uses 
        # a different behavior at train time and test time.
        self.input_tensors.append(K.learning_phase())

        self.vis_node_tensors = []
        for i in self.vis_nodes:
            self.vis_node_tensors.append(i)
        for i in self.extra_nodes:
            self.vis_node_tensors.append(i)
        self.vis_node_tensors.append(K.learning_phase())
        
        #If outputchanels are specified, use it.
        #Otherwise evalueate all outputs.
        self.outchannels = outchannels
        if len(self.outchannels) == 0: 
            if verbose: print("Evaluated output channel (0-based index): All")
            if K.backend() == "tensorflow":
                self.outchannels = range(self.model.output.shape[1]._value)
            elif K.backend() == "theano":
                self.outchannels = range(self.model.output._keras_shape[1])
        else:
            if verbose: 
                print("Evaluated output channels (0-based index):")
                print(','.join([str(i) for i in self.outchannels]))

        #Build evaluate functions for visual nodes.
        if self.vis_nodes != self.model.inputs:
            self.get_vis_values = K.function( 
                inputs=self.input_tensors, 
                outputs=self.vis_nodes)
        else:
            self.get_vis_values = lambda x: x[:-1]

        #Build gradient functions for desired output channels.
        self.get_gradients = {}
        if verbose: print("Building gradient functions")
        
        # Evaluate over all requested channels.
        for c in self.outchannels:
            # Get tensor that calculates gradient
            if K.backend() == "tensorflow":
                gradients = self.model.optimizer.get_gradients(
                                            self.model.output[:, c], self.vis_nodes)
            if K.backend() == "theano":
                gradients = self.model.optimizer.get_gradients(
                                      self.model.output[:, c].sum(), self.vis_nodes)
                
            # Build computational graph that computes the tensors given inputs
            self.get_gradients[c] = K.function( inputs=self.vis_node_tensors, 
                                                outputs=gradients)
            
            # This takes a lot of time for a big model with many tasks.
            # So lets print the progress.
            if verbose:
                sys.stdout.write('\r')
                sys.stdout.write("Progress: "+str(int((c+1)*1.0/len(self.outchannels)*1000)*1.0/10)+"%")
                sys.stdout.flush()
        # Done
        if verbose: print("\nDone.")
            
                
    '''
    Input: sample to explain, channel to explain
    Optional inputs:
        - reference: reference values (defaulted to 0s).
        - steps: # steps from reference values to the actual sample (defualted to 50).
    Output: list of numpy arrays to integrated over.
    '''
    def explain(self, sample_input, outc=0, reference_input=False, num_steps=50, extra_values=[], verbose=0):

        if isinstance(sample_input, np.ndarray):
            sample_input = [sample_input]        

        if isinstance(sample_input, list):
            sample = []
            reference = []
            #If reference is present, reference and sample size need to be equal.
            if reference_input == False:
                reference_input = []
                for idx, si in enumerate(sample_input):
                    reference_input.append(np.zeros(si.shape, dtype=si.dtype))
            else:
                assert len(sample_input) == len(reference_input)
            
            sample = self.get_vis_values(sample_input + [0])
            reference = self.get_vis_values(reference_input + [0])
            print(len(sample))
            print(len(reference))
        
        # Each element for each input stream.
        samples = []
        numsteps = []
        step_sizes = []
        
        # If multiple inputs are present, feed them as list of np arrays. 
        if isinstance(sample, list):
            assert len(sample) == len(reference)
            for i in range(len(sample)):
                _output = integrated_gradients.linearly_interpolate(
                                    sample[i], reference[i], num_steps)
                samples.append(_output[0])
                numsteps.append(_output[1])
                step_sizes.append(_output[2])
        
        # Desired channel must be in the list of outputchannels
        assert outc in self.outchannels
        if verbose: print("Explaning the "+str(self.outchannels[outc])+"th output.")
            
        # For tensorflow backend
        _input = []
        for s in samples:
            _input.append(s)
        for s in extra_values:
            _input.append(s)
        _input.append(0)
        
        if K.backend() == "tensorflow": 
            gradients = self.get_gradients[outc](_input)
        elif K.backend() == "theano":
            gradients = self.get_gradients[outc](_input)
            if len(self.model.inputs) == 1:
                gradients = [gradients]
        
        explanation = []
        for i in range(len(gradients)):
            _temp = np.sum(gradients[i], axis=0)
            explanation.append(np.multiply(_temp, step_sizes[i]))
           
        # Format the return values according to the input sample.
        if isinstance(sample, list):
            return explanation
        elif isinstance(sample, np.ndarray):
            return explanation[0]
        return -1

    
    '''
    Input: numpy array of a sample
    Optional inputs:
        - reference: reference values (defaulted to 0s).
        - steps: # steps from reference values to the actual sample.
    Output: list of numpy arrays to integrate over.
    '''
    @staticmethod
    def linearly_interpolate(sample, reference=False, num_steps=50):
        # Use default reference values if reference is not specified
        if reference is False: reference = np.zeros(sample.shape);

        # Reference and sample shape needs to match exactly
        assert sample.shape == reference.shape

        # Calcuated stepwise difference from reference to the actual sample.
        ret = np.zeros(tuple([num_steps] +[i for i in sample.shape]))
        for s in range(num_steps):
            ret[s] = reference+(sample-reference)*(s*1.0/num_steps)

        return ret, num_steps, (sample-reference)*(1.0/num_steps)

