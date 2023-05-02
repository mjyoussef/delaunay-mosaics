import cv2
import numpy as np
from utils.sampling import randomly_sample_points, sparsify

'''
Baseline Delaunay Triangulation...

Step 1: Randomly sample points from the image

Step 2: Pass those points as input to the triangulation algorithm to get all of the 
triangles and edges

Step 3: Update the color of every pixel in a triangle to the average color in that triangle

Step 4: Save and display the mosaic from step 3

'''

def super_triangle(points):
    pass

def triangulate(points):
    pass



################### Arg parse function ######################

def display_mosaic_baseline(path):
    '''
    Displays a Delaunay mosaic of the image

    path: path to image

    returns void
    '''
    image = cv2.imread(path)
    points = randomly_sample_points(image, 1000)
    points = sparsify(points, 30)

    pass