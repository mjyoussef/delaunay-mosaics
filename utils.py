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

def orientation_of(p1, p2, p3):
    '''
    computes the orientation of three points

    p1, p2, p3: coordinates of points

    returns -1 for cw, 0 for colinear, and 1 for ccw
    '''
    det_matrix = np.array([[p1[0], p1[1], 1], [p2[0], p2[1], 1], [p3[0], p3[1], 1]])
    det_sign = np.linalg.det(det_matrix)
    if (det_sign < 0):
        det_sign = -1
    
    if (det_sign > 0):
        det_sign = 1

    return det_sign

def segments_intersect(e1, e2):
    '''
    determines if two edges intersect each other
    ** Note: edges with a single matching endpoint do not intersect for our purposes **

    e1, e2: edges

    returns true/false
    '''
    if (e1[0] == e2[0] or e1[0] == e2[1] or e1[1] == e2[0] or e1[1] == e2[1]):
        return False
    
    o_1 = orientation_of(e1[0], e1[1], e2[0])
    o_2 = orientation_of(e1[0], e1[1], e2[1])

    o_3 = orientation_of(e2[0], e2[1], e1[0])
    o_4 = orientation_of(e2[0], e2[1], e1[1])

    if (o_1 == 0 and o_2 == 0 and o_3 == 0 and o_4 == 0): # colinear

        x_projs = [(e1[0][0], 0), (e1[1][0], 0), (e2[0][0], 1), (e2[1][0], 1)]
        list.sort(x_projs, key=lambda x: x[0])

        y_projs = [(e1[0][1], 0), (e1[1][1], 0), (e2[0][1], 1), (e2[1][1], 1)]
        list.sort(y_projs, key=lambda y: y[0])

        # check if there is overlap in the x, y projections of the segments
        return x_projs[0][1] != x_projs[1][1] or y_projs[0][1] != y_projs[0][1]
    
    # if points are not colinear, they intersect when
    # the orientations 1,2 and 3,4 are different
    return o_1 != o_2 and o_3 != o_4

def filter_edges(edges, adj_mat):
    '''
    Helper function for finding a maximal independent subset in the adj_mat
    ** Note: this function is attemtping to find something as close as possible to the 
    maximum independent subset given that the adjaceny matrix tends to be very sparse **

    edges: a list of edges
    adj_mat: an adjacency matrix where the "nodes" are edges from the `edges` list
    and an edge between two nodes exists if they intersect each other

    returns a subset of edges from `edges` s.t that no edge intersects another edge
    '''
    n = len(edges)
    
    # counts the number of edges each edge neighbors in adj_mat
    intersects = adj_mat.sum(axis=1)

    # keep track of how many edges need to be removed
    intersects_copy = intersects.copy()
    intersects_copy[intersects_copy > 0] = 1
    remaining = intersects_copy.sum()

    # indices of edges that need to be removed
    edges_to_remove = set()

    while (remaining > 0):
        # find edge with greatest number of neighbors
        edge_idx = np.argmax(intersects)
        edges_to_remove.add(edge_idx)

        # iterate over neighbors, updating `remaining` along the way
        for i in range(n):
            if (adj_mat[edge_idx][i] == 1):
                intersects[i] -= 1
                if (intersects[i] == 0):
                    remaining -= 1
        
        intersects[edge_idx] = 0
        remaining -= 1
    
    new_edges = []
    for i in range(n):
        if (i not in edges_to_remove):
            new_edges.append(edges[i])
    
    return new_edges

def remove_intersecting_edges(edges):
    '''
    removes edges that intersect at anything other than end-points

    Note: the goal of this procedure is to remove as few edges as possible
    such that none of the edges in the new list intersect; this is impossible to do efficiently
    since you can reduce maximum independent sets to this problem
    '''
    n = len(edges)
    adj_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(i):
            if (segments_intersect(edges[i], edges[j])):
                adj_mat[i][j] = 1
                adj_mat[j][i] = 1
    
    # remove edge that intersects the greatest number of edges
    return filter_edges(edges, adj_mat)

def distance_less_thresh(e1, e2, dist_thresh, theta_thresh):
    '''
    computes distance between two edges

    Note: this is a custom distance function that considers two edges
    as being close if the distance between them is below a threshold AND
    the angular distance between them is below a threshold
    '''

    x1, y1 = e1
    x2, y2 = e2
    ep_dist = min(dist(x1, x2), dist(x1, y2), dist(y1, x2), dist(y1, y2))

    if (ep_dist >= dist_thresh):
        return False
    
    v1, v2 = None, None
    if (ep_dist == dist(x1, x2)):
        y1[0] -= x1[0]
        y1[1] -= x1[1]
        y2[0] -= x2[0]
        y2[1] -= x2[1]
        v1, v2 = y1, y2
    elif (ep_dist == dist(x1, y2)):
        y1[0] -= x1[0]
        y1[1] -= x1[1]
        x2[0] -= y2[0]
        x2[1] -= y2[1]
        v1, v2 = y1, x2
    elif (ep_dist == dist(y1, x2)):
        x1[0] -= y1[0]
        x1[1] -= y1[1]
        y2[0] -= x2[0]
        y2[1] -= x2[1]
        v1, v2 = x1, y2
    else:
        x1[0] -= y1[0]
        x1[1] -= y1[1]
        x2[0] -= y2[0]
        x2[1] -= y2[1]
        v1, v2 = x1, x2
    
    # compute cos similarity
    dot_prod = (v1[0] * v2[0]) + (v1[1] * v2[1])
    cos_sim = dot_prod / (dist(v1, (0,0)) * dist(v2, (0,0)))

    return np.arccos(cos_sim) < theta_thresh


def remove_close_edges(edges, dist_thresh, theta_thresh):
    '''
    removes edges that are too close to one another
    '''
    n = len(edges)
    adj_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(i):
            if (distance_less_thresh(edges[i], edges[j], dist_thresh, theta_thresh)):
                adj_mat[i][j] = 1
                adj_mat[j][i] = 1

    return filter_edges(edges, adj_mat)


def add_entries_in_radius(img, i, j, new_parent, radius, q):
    '''
    shuffles and adds elements from a (square) radius around i,j to the queue
    '''

    entries = []
    for i_hat in range(max(0, i-radius), min(img.shape[0], i+radius+1)):
        for j_hat in range(max(0, j-radius), min(img.shape[1], j+radius+1)):
            entries.append((i_hat, j_hat, new_parent))
    
    random.shuffle(entries)

    for entry in entries:
        q.put(entry)

def search(img, i_start, j_start, min_length, edges, radius):
    '''
    Helper method for `sample_edges_from_edges`
    '''
    q = Queue()

    # stores coordinate, reference to parent

    q.put((i_start, j_start, (i_start, j_start)))

    while (not q.empty()):
        i, j, parent = q.get()

        if (i < 0 or i >= img.shape[0] or j < 0 or j >= img.shape[1]):
            continue

        if (img[i][j] == 0):
            continue

        new_parent = parent

        # check distance to parent
        i2, j2 = parent
        d = dist((i, j), (i2, j2))

        if (d >= min_length):
            edges.append(((i, j), (i2, j2)))
            new_parent = (i, j)
        
        img[i][j] = 0

        # add neighboring points (within a certain radius) to the queue
        add_entries_in_radius(img, i, j, new_parent, radius, q)
        
def sample_edges_from_edges(img, min_length, radius=1):
    '''
    Randomly samples edges that are bounded by certain length threshold

    img: image that has already been run through an edge detector
    min_length: minimum edge length (L2 distance)
    max_length: maximum edge length (L2 distance)

    returns edges in the image as a list of pairs of tuples
    '''

    edges = []
    img_copy = img.copy()
    for i in range(img_copy.shape[0]):
        for j in range(img_copy.shape[1]):
            if (img_copy[i][j] != 0):
                search(img_copy, i, j, min_length, edges, radius)
    
    return remove_intersecting_edges(edges)

if __name__ == '__main__':
    blurred, output = get_edges("images/portraits/abe.jpeg", 7, 7, 10, 150)
    edges = sample_edges_from_edges(output, 40, radius=1)

    width, height = output.shape
    image = np.ones((width, height)) * 255

    for p1, p2 in edges:
        cv2.line(image, (p1[1], p1[0]), (p2[1], p2[0]), (0, 255, 0), thickness=2)
    
    cv2.imshow("output", image)
    cv2.waitKey(0)