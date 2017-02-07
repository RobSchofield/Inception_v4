#!/usr/bin/python
'''
Data manager loads, pre-processes, stores images, and stores processed images.

Author: Robert Schofield
Last Modified: 02/07/2017
'''

import numpy
import pandas
import inception_v4

import matplotlib.pyplot as plt

from PIL import Image


class Data(object):
    def __init__(self, file_names=[], data_dir=''):
        '''
        Initialize variables.
        '''
        self.file_names = file_names
        self.data_dir = data_dir
        self.images = None
        self.labels = None
        self.n_images = 0
        self.v_rep = None

    def preprocess_image(self, image):
        '''
        Preprocess image by normalizing all pixel values to
        a range between -1.0 and 1.0.
        '''
        image = numpy.array(image)
        image = numpy.divide(image, 255.0)
        image = numpy.subtract(image, 1.0)
        image = numpy.multiply(image, 2.0)
        return image

    def import_images(self):
        '''
        Import images.
        '''
        print('Importing images...')
        self.n_images = len(self.file_names)
        self.images = numpy.ones(shape=(self.n_images, 299, 299, 3),
                                 dtype=numpy.float32)
        for _, file_name in enumerate(self.file_names):
            try:
                image = Image.open(self.data_dir + file_name).resize((299, 299))
                self.images[_] = self.preprocess_image(image)
            except:
                print(file_name + ' invalid')
        print('Sucessfully imported %d images.' % (self.n_images))
        return None
    
    def import_labels(self, file_name):
        '''
        Import labeled data.
        '''
        print('Importing labeled data...')
        df = pandas.read_csv(self.data_dir + file_name, index_col=0)
        self.labels = df.values
        print('Sucess.')
        return None

    def import_data(self, file_name):
        '''
        Import the vector representation of a csv file.
        '''
        print('Importing data...')
        df = pandas.read_csv(self.data_dir + file_name, index_col=0)
        self.v_rep = df.values
        print('Sucess.')
        return None

    def export_data(self, file_name):
        '''
        Export the vector representation to csv file.
        '''
        print('Exporting data...')
        df = pandas.DataFrame(self.v_rep, index=self.file_names)
        df.to_csv(self.data_dir + file_name)
        print('Sucess.')
        return None

if __name__ == '__main__':
    print('datamanager.py')
