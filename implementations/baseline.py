import cv2
import numpy as np
from utils.sampling import randomly_sample_points, sparsify
from utils.geo import Segment, Triangle

'''
Baseline Delaunay Triangulation...

Step 1: Randomly sample points from the image

Step 2: Pass those points as input to the triangulation algorithm to get all of the 
triangles and edges

Step 3: Update the color of every pixel in a triangle to the average color in that triangle

Step 4: Save and display the mosaic from step 3

'''

def super_triangle(points):
    min_x = min_y = np.Inf
    max_x = max_y = -np.Inf

    for pt in points:
        min_x = min(min_x, pt[0])
        min_y = min(min_y, pt[1])
        max_x = max(max_x, pt[0])
        max_y = max(max_y, pt[1])

    dx = (max_x - min_x) * 10
    dy = (max_y - min_y) * 10

    p0 = (min_x - dx, min_y - (dy*3))
    p1 = (min_x - dx, max_y + dy)
    p2 = (max_x + (3*dx), max_y + dy)

    return Triangle(p0, p1, p2)

def add_opposite_triangle(pt, t, stack, segments_to_triangle_dict):
    opp_segment = t.opposite_segment(pt)
    tris_w_opp_segment = segments_to_triangle_dict.get(opp_segment)

    if (len(tris_w_opp_segment) > 1):
        opposite_triangle = tris_w_opp_segment[0]
        if (opposite_triangle == t):
            opposite_triangle = tris_w_opp_segment[1]
        
        stack.append((opposite_triangle, opp_segment))

def insert_point(pt, triangles, segments_to_triangle_dict):
    init_t = None
    stack = []

    for t in triangles:
        if (t.encloses(pt)):
            init_t = t
            break
    
    new_tri1 = Triangle(pt, init_t.A_coords, init_t.B_coords)
    new_tri2 = Triangle(pt, init_t.A_coords, init_t.C_coords)
    new_tri3 = Triangle(pt, init_t.B_coords, init_t.C_coords)

    add_opposite_triangle(pt, new_tri1, stack, segments_to_triangle_dict)
    add_opposite_triangle(pt, new_tri2, stack, segments_to_triangle_dict)
    add_opposite_triangle(pt, new_tri3, stack, segments_to_triangle_dict)

    while (len(stack) > 0):
        t, opp_segment = stack.pop()
        if (t.in_circumcircle(pt)): # swap diagonals of the convex quadrilateral

            tris_w_opp_segment = segments_to_triangle_dict.get(opp_segment)
            opp_tri_segments0 = tris_w_opp_segment[0].non_opposite_segments(pt, opp_segment)
            opp_tri_segments1 = tris_w_opp_segment[1].non_opposite_segments(pt, opp_segment)

            # find edges sharing similar endpoints from opp_segment
            left, right = [], []
            if (opp_tri_segments0[0].contains(opp_segment.p1)):
                left.append(opp_tri_segments0[0])
            else:
                left.append(opp_tri_segments0[1])
                right.append(opp_tri_segments0[0])
            
            if (opp_tri_segments1[0].contains(opp_segment.p1)):
                left.append(opp_tri_segments1[0])
            else:
                left.append(opp_tri_segments1[1])
                right.append(opp_tri_segments1[0])

            # remove both triangles in `tris_w_opp_segment` from `triangles`
            # dictionary update: remove `opp_segment` from dictionary and update values associated w/ segments from 
            # `opp_tri_segments0` and `opp_tri_segments1`
            

            # add two new triangles to triangles
            # dictionary update: update values associated with segments from `opp_tri_segments0` and `opp_tri_segments1`
            # and add in the new diagonal segment

            # update the stack using the two new triangles


def triangulate(points):

    st = super_triangle(points)
    triangles = [st]

    # maps edges to corresponding triangles (2 max)
    # note: useful for performing Lawson's search
    segments_to_triangle_dict = {st.a: [st], st.b: [st], st.c: [st]}

    for pt in points:
        triangles = insert_point(pt, triangles, segments_to_triangle_dict)
    
    return triangles
        


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