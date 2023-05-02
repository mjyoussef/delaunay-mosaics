import cv2
import numpy as np
from queue import Queue
import random
from geo import dist, segments_intersect


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

    return image, blurred, output

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
    return remove_close_edges(non_intersecting_edges, dist_thresh, theta_thresh)

if __name__ == '__main__':
    dist_thresh = 30
    theta_thresh = np.pi/6

    blurred, output = get_edges("images/portraits/abe.jpeg", 5, 5, 25, 80)
    edges = sample_edges_from_edges(output, 30, dist_thresh, theta_thresh, radius=1)

    width, height = output.shape
    image = np.ones((width, height, 3)) * 255

    for p1, p2 in edges:
        cv2.line(image, (p1[1], p1[0]), (p2[1], p2[0]), (0, 255, 0), thickness=2)
        cv2.circle(image, (p1[1], p1[0]), 2, (255, 0, 0), thickness=-1)
        cv2.circle(image, (p2[1], p2[0]), 2, (255, 0, 0), thickness=-1)
    
    cv2.imshow("output", image)
    cv2.waitKey(0)