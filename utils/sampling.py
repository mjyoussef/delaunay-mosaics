import sys
sys.path.append("../../delaunay-mosaics")

import cv2
import numpy as np
from queue import Queue
import random
from utils.geo import dist, segments_intersect, Segment


########################## Helpers ###############################
def MIS(vertices, adj_mat):
    '''
    Helper function for finding a maximal independent subset in the adj_mat
    ** Note: this function is attemtping to find something as close as possible to the 
    maximum independent subset given that the adjaceny matrix tends to be very sparse **

    edges: a list of edges
    adj_mat: an adjacency matrix where the "nodes" are edges from the `edges` list
    and an edge between two nodes exists if they intersect each other

    returns a subset of edges from `edges` s.t that no edge intersects another edge
    '''
    n = len(vertices)
    
    # counts the number of vertices each vertex neighbors in adj_mat
    intersects = adj_mat.sum(axis=1)

    # keep track of how many vertices need to be removed
    intersects_copy = intersects.copy()
    intersects_copy[intersects_copy > 0] = 1
    remaining = intersects_copy.sum()

    # indices of edges that need to be removed
    vertices_to_remove = set()

    while (remaining > 0):
        # find edge with greatest number of neighbors
        vertex_idx = np.argmax(intersects)
        vertices_to_remove.add(vertex_idx)

        # iterate over neighbors, updating `remaining` along the way
        for i in range(n):
            if (adj_mat[vertex_idx][i] == 1):
                intersects[i] -= 1
                if (intersects[i] == 0):
                    remaining -= 1
        
        intersects[vertex_idx] = 0
        remaining -= 1
    
    new_vertices = []
    for i in range(n):
        if (i not in vertices_to_remove):
            new_vertices.append(vertices[i])
    
    return new_vertices

def sparsify(points, dist_threshold):
    '''
    Filters points such that no two points are within a distance threshold

    points: list of x,y coordinates
    dist_threshold: Euclidean distance threshold

    returns a filtered list of points
    '''

    n = len(points)
    adj_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            if (dist(points[i], points[j]) < dist_threshold):
                adj_mat[i][j] = 1
                adj_mat[j][i] = 1
    
    return MIS(points, adj_mat)

######################### Utilities ################################## 

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
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=sigmaX, sigmaY=sigmaY)
    output = cv2.Canny(blurred, thresh1, thresh2)

    return image, blurred, output

def randomly_sample_points(img, cnt, qualityScore, minDistance, mode="corner"):
    '''
    Randomly samples points in an image

    img: untouched input image
    cnt: number of points to sample
    qualityScore: qualityScore for corner detection
    minDistance: minimum distance between points
    num_points: the number of points to sample

    returns a list of tuples representing the randomly sampled points
    '''
    if mode=="corner":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, cnt, qualityScore, minDistance)
        corners = np.squeeze(corners)

        coords = []
        for corner in corners:
            coords.append((int(corner[0]), int(corner[1])))
        
        return coords
    else:
        x = np.random.randint(1, img.shape[0]-1, (cnt, 1))
        x = np.squeeze(x)
        y = np.random.randint(1, img.shape[1]-1, (cnt, 1))
        y = np.squeeze(y)

        coords = []
        for idx in range(x.shape[0]):
            coords.append((int(y[idx]), int(x[idx])))

        return sparsify(coords, minDistance)

def sample_points_from_edges(img, cnt, minDistance):
    '''
    Randomly samples points along the edges of an image

    img: image that has already been run through an edge detector
    cnt: number of points to sample
    minDistance: minimum distance threshold

    returns points in the image as a list of tuples
    '''

    coords = np.column_stack(np.where(img > 0))

    samples = coords[np.random.choice(coords.shape[0], cnt, replace=False), :]
    samples = np.squeeze(samples)

    coords = []
    for idx in range(samples.shape[0]):
        coords.append((samples[idx][1], samples[idx][0]))

    return sparsify(coords, minDistance)

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
    return MIS(edges, adj_mat)

def distance_less_thresh(e1, e2, dist_thresh, theta_thresh):
    '''
    determines if two edges are considered close
    ** Note: this is a custom distance function that considers two edges
    as being close if the distance between them is below a threshold AND
    the angular distance between them is below a threshold **

    e1, e2: edges
    dist_thresh: distance threshold
    theta_thresh: angular distance threshold

    returns true/false
    '''

    x1, y1 = e1
    x2, y2 = e2
    ep_dist = min(dist(x1, x2), dist(x1, y2), dist(y1, x2), dist(y1, y2))

    if (ep_dist >= dist_thresh):
        return False
    
    # vectors representing the two edges
    v1, v2 = None, None
    if (ep_dist == dist(x1, x2)):
        v1 = (y1[0]-x1[0], y1[1]-x1[1])
        v2 = (y2[0]-x2[0], y2[1]-x2[1])
    elif (ep_dist == dist(x1, y2)):
        v1 = (y1[0]-x1[0], y1[1]-x1[1])
        v2 = (x2[0]-y2[0], x2[1]-y2[1])
    elif (ep_dist == dist(y1, x2)):
        v1 = (x1[0]-y1[0], x1[1]-y1[1])
        v2 = (y2[0]-x2[0], y2[1]-x2[1])
    else:
        v1 = (x1[0]-y1[0], x1[1]-y1[1])
        v2 = (x2[0]-y2[0], x2[1]-y2[1])
    
    # compute cos similarity
    dot_prod = (v1[0] * v2[0]) + (v1[1] * v2[1])
    cos_sim = dot_prod / (dist(v1, (0,0)) * dist(v2, (0,0)))
    cos_sim = np.clip(cos_sim, -1, 1)

    return np.arccos(cos_sim) < theta_thresh

def remove_close_edges(edges, dist_thresh, theta_thresh):
    '''
    removes edges that are too close to one another

    edges: a list of edges
    dist_thresh: distance threshold
    theta_thresh: angular distance threshold

    returns a list of edges s.t that no two edges are "close" to one another
    '''
    n = len(edges)
    adj_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(i):
            if (distance_less_thresh(edges[i], edges[j], dist_thresh, theta_thresh)):
                adj_mat[i][j] = 1
                adj_mat[j][i] = 1

    return MIS(edges, adj_mat)


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
        
def sample_edges_from_edges(img, min_length, dist_thresh, theta_thresh, radius=1):
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
    
    non_intersecting_edges = remove_intersecting_edges(edges)
    output = remove_close_edges(non_intersecting_edges, dist_thresh, theta_thresh)

    final_output = []
    for edge in output:
        final_output.append(((edge[0][1], edge[0][0]), (edge[1][1], edge[1][0])))
    return final_output



########################### Post processing utilities #########################
def edges_to_segments(edges):
    '''
    Converts a list of pairs of points into segment objects

    edges: a list of edges

    returns a list of segment objects
    '''
    segments = []
    for e in edges:
        segments.append(Segment(e[0], e[1]))
    
    return segments

def add_pts_to_pts(img, pts, count, dist_threshold):
    '''
    Adds supplementary points to the points while maintaining the required distance threshold

    pts: points
    count: num points
    dist_threshold: distance threshold

    returns a list of additional points
    '''

    new_pts_raw = randomly_sample_points(img, count, 0, dist_threshold, mode="not corner")
    new_pts_final = []
    for pt in new_pts_raw:
        withinThresh = True
        for pt2 in pts:
            if (dist(pt, pt2) < dist_threshold):
                withinThresh = False
                break
        if (withinThresh):
            new_pts_final.append(pt)
    
    return new_pts_final

def add_pts_to_segments(img, segments, count, dist_threshold):
    '''
    Adds supplementary points to the segments while maintaining the required distance threshold

    segments: a list of segments
    count: num points
    dist_threshold: distance threshold

    returns a list of additional points
    '''

    seg_pts = set([])
    for seg in segments:
        seg_pts.add(seg.p1)
        seg_pts.add(seg.p2)
    
    out = add_pts_to_pts(img, seg_pts, count, dist_threshold)
    return out