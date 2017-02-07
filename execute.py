#!/usr/bin/python
'''
This program reads in a file with 

Author: Robert Schofield
Last Modified: 02/07/2017
'''

import datamanager
import neuralnetworkmanager
import numpy
import pandas

file_path = 'C:\\Users\\admin\\Documents\\Uptown Treehouse\\API\\WUIndie_data\\' # path to where the images are located
file_name = 'WUIndie_data.csv' # path to where .csv files with labels and file names
data = pandas.read_csv(file_path + file_name, index_col=0) # open csv file

# all the above was to obtain the file names of the pictures
file_names = data.index.values

# create a data manager object to handle all of the data
dm = datamanager.Data(file_names, file_path)
# create a neural network manager object to handle the neural networks
nnm = neuralnetworkmanager.NeuralNetworkManager(dm, True)

# import the images
dm.import_images()

# analyze the images
nnm.analyze_images()

# save file to csv
nnm.export_data('file.csv')
