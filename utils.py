import cv2
import numpy as np
from queue import Queue
import random


def get_edges(path, sigmaX, sigmaY, thresh1, thresh2):
    '''
    Extracts edges from images

    sigmaX: x variance for Gaussian filter
    sigmaY: y variance for Gaussian filter
    thresh1: lower threshold for hysteris
    thresh2: upper threshold for hysteris

    returns blurred output and output w/ outlined edges
    '''
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (sigmaX, sigmaY), 0)
    output = cv2.Canny(blurred, thresh1, thresh2)

    # cv2.imshow("blurred", blurred)
    # cv2.imshow("output", output)
    # cv2.waitKey(0)

    return blurred, output

def randomly_sample_points(img, num_points):
    '''
    Randomly samples points in an image

    img: untouched input image
    num_points: the number of points to sample

    returns a list of tuples representing the randomly sampled points
    '''
    pass

def sample_points_from_edges(img, num_points):
    '''
    Randomly samples points along the edges of an image

    img: image that has already been run through an edge detector
    num_points: number of points to sample

    returns points in the image as a list of tuples
    '''
    
    pass


def dist(p1, p2):
    return np.sqrt( ((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2) )

def search(img, i_start, j_start, min_length, max_length, edges):
    q = Queue()

    # stores coordinate, reference to parent, # number of times an 
    # option of adding a coordinate has been rejected

    q.put((i_start, j_start, None, 0))

    while (not q.empty()):
        i, j, parent, rejected = q.get()

        if (img[i][j] == 0):
            continue


        new_parent = parent
        new_rejected = rejected+1

        # check distance to parent
        if (parent):
            i2, j2 = parent
            dist = dist((i, j), (i2, j2))

            if (dist > min_length):
                # w/ probability 1 / (max_length - min_length - rejected)
                # form a new edge between the parent and the current point
                points_left = max_length - min_length - rejected
                if (points_left == 0):
                    edges.append(((i, j), (i2, j2)))
                    new_parent = (i, j)
                    new_rejected = 0
                else:
                    rand_num = random.uniform(0, 1)
                    if (rand_num <= (1. / points_left)): # add edge
                        edges.append(((i, j), (i2, j2)))
                        new_parent = (i, j)
                        new_rejected = 0
            else:
                if (max_length - min_length <= rejected):
                    continue
        
        img[i][j] = 0

        # add neighboring points to the queue
        q.put((i-1, j, new_parent, new_rejected))
        q.put((i+1, j, new_parent, new_rejected))
        q.put((i, j-1, new_parent, new_rejected))
        q.put((i, j+1, new_parent, new_rejected))

    return edges
        

def sample_edges_from_edges(img, min_length, max_length):
    '''
    Randomly samples edges that are bounded by certain length threshold

    img: image that has already been run through an edge detector
    min_length: minimum edge length (L2 distance)
    max_length: maximum edge length (L2 distance)

    returns edges in the image as a list of pairs of tuples
    '''

    edges = []
    img_copy = img.clone()
    for i in range(img_copy.shape[0]):
        for j in range(img_copy.shape[1]):
            if (img_copy[i][j] != 0):
                search(img_copy, i, j, min_length, max_length, edges)
    
    return edges

if __name__ == '__main__':
    get_edges("images/portraits/abe.jpeg", 7, 7, 30, 150)