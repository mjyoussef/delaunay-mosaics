import numpy as np
import cv2

'''
Utility functions for coloring in the triangulation
'''

def fill_triangles(image, triangles):
    '''
    Fills in each triangle of the image w/ its average color

    image: image (gets changed in place)
    triangles: an n x 3 array of triangle points

    returns void
    '''

    for triangle in triangles:
        mask = np.zeros_like(image[:,:,0])

        # the region within the triangle is set to 255
        cv2.drawContours(mask, np.array([triangle]), 0, 255, -1)

        # compute average color and fill in triangle
        B_mean, G_mean, R_mean, _ = cv2.mean(image, mask=mask)
        cv2.fillPoly(image, np.array([triangle]), (B_mean, G_mean, R_mean))