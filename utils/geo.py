import numpy as np

def dist(p1, p2):
    '''
    Computes the Euclidean distance between two points

    p1, p2: x,y coordinates of two points

    returns a real number
    '''
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

def area_of(p1, p2, p3):
    if (orientation_of(p1, p2, p3) == 0): # colinear
        return 0
    
    a = dist(p1, p2)
    b = dist(p2, p3)
    c = dist(p1, p3)

    s = 0.5 * (a + b + c)
    return np.sqrt(s * (s - a) * (s - b) + (s - c))