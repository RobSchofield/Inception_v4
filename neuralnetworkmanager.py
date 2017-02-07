#!/usr/bin/python
'''
Implement Inception-ResNet-V2 trained with LSVRC2012 that takes in
an image and returns the probability of each class or an abstract
1536-dimensional vector representation.

Author: Robert Schofield
Last Modified: 02/07/2017
'''

import inception_v4
import numpy


class NeuralNetworkManager(object):

    def __init__(self, data=None, abstract_vector=False):
        '''
        Initialize variables.
        '''
        self.data = data
        self.cnn = inception_v4.create_model(weights_path="pretrained.h5",
                                             abstract=abstract_vector)

    def analyze_images(self):
        '''
        Process all images with the neural networks.
        '''
        print('Analyzing images with neural network...')
        self.data.v_rep = self.cnn.predict(self.data.images, verbose=1)
        print('Sucessfully analyzed %d images.' % (self.data.n_images))
        return None

if __name__ == '__main__':
    print('neuralnetworkmanager.py')
