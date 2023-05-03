import sys
sys.path.append("../../delaunay-mosaics")

import cv2
import numpy as np
from utils.geo import Triangle
from utils.sampling import randomly_sample_points, sample_points_from_edges, get_edges, sparsify
from utils.coloring import fill_triangles

'''
Baseline Delaunay Triangulation...

Step 1: Randomly sample points from the image

Step 2: Pass those points as input to the triangulation algorithm to get all of the 
triangles and edges

Step 3: Update the color of every pixel in a triangle to the average color in that triangle

Step 4: Save and display the mosaic from step 3

'''

def super_triangle(points):
    '''
    Creates a super triangle that encloses all of the points

    points: a list of x,y coordinates

    returns a triangle object
    '''
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
    '''
    Updates stack of triangles given a point and a triangle containing that point; 
    see Lawson's algorithm for more info

    pt: x,y coordinates
    t: a triangle object
    stack: a stack of triangle objects w/ opposite segments
    segments_to_triangle_dict: mapping from segments to triangles

    returns void
    '''
    opp_segment = t.opposite_segment(pt)
    tris_w_opp_segment = segments_to_triangle_dict.get(opp_segment)
    opposite_triangle = None
    for t_w_opp in tris_w_opp_segment:
        if (t_w_opp != t):
            opposite_triangle = t_w_opp
            break
    
    if (opposite_triangle != None):
        stack.append((opposite_triangle, opp_segment))

def triangle_from_segments(segments):
    '''
    Constructs a triangle given a list of segments that form that triangle

    segments: a list of segment objects

    returns a triangle object
    '''
    pts = []
    for seg in segments:
        if (pts.count(seg.p1) == 0):
            pts.append(seg.p1)
        if (pts.count(seg.p2) == 0):
            pts.append(seg.p2)
    
    if (len(pts) != 3):
        raise Exception("A triangle must have exactly 3 points")
    
    p1, p2, p3 = pts
    return Triangle(p1, p2, p3)

def update_dict(seg, triangle, segments_to_triangle_dict):
    '''
    Updates the segments to triangle dictionary with a seg and its corresponding triangle

    seg: a segment object
    triangle: a triangle object
    segments_to_triangle_dict: segments dictionary

    returns void
    '''
    if (segments_to_triangle_dict.get(seg) == None):
        segments_to_triangle_dict.update({seg: [triangle]})
    else:
        segments_to_triangle_dict.get(seg).append(triangle)

def insert_point(pt, triangles, segments_to_triangle_dict):
    '''
    Subroutine for Lawson's search algorithm, which updates the
    triangulation for a given input point

    pt: x,y coordinates
    triangles: a list of triangle objects
    segments_to_triangle_dict: segments dictionary

    returns void
    '''
    init_t = None
    stack = []

    for t in triangles:
        if (t.encloses(pt)):
            init_t = t
            break
    
    # break apart the enclosing triangle into 3 new triangles
    new_tri1 = Triangle(pt, init_t.A_coords, init_t.B_coords)
    new_tri2 = Triangle(pt, init_t.A_coords, init_t.C_coords)
    new_tri3 = Triangle(pt, init_t.B_coords, init_t.C_coords)

    # remove old triangle from dictionary and from `triangles`
    segments_to_triangle_dict.get(init_t.a).remove(init_t)
    segments_to_triangle_dict.get(init_t.b).remove(init_t)
    segments_to_triangle_dict.get(init_t.c).remove(init_t)
    triangles.remove(init_t)

    # add opposite triangle to dictionary and to triangles
    add_opposite_triangle(pt, new_tri1, stack, segments_to_triangle_dict)
    add_opposite_triangle(pt, new_tri2, stack, segments_to_triangle_dict)
    add_opposite_triangle(pt, new_tri3, stack, segments_to_triangle_dict)

    # add new line segments + references to new triangle in dictionary and update `triangles`
    new_triangles = [new_tri1, new_tri2, new_tri3]
    for t in new_triangles:
        triangles.append(t)
        update_dict(t.a, t, segments_to_triangle_dict)
        update_dict(t.b, t, segments_to_triangle_dict)
        update_dict(t.c, t, segments_to_triangle_dict)
    
    while (len(stack) > 0): # see Lawson's algorithm

        # triangle opposite to `pt`
        t, opp_segment = stack.pop()

        if (t.in_circumcircle(pt)): # swap diagonals of the convex quadrilateral

            tris_w_opp_segment = segments_to_triangle_dict.get(opp_segment).copy()
            for t1 in tris_w_opp_segment:
                if (not t1.contains_segment(opp_segment)):
                    raise Exception("Found a triangle not containing opposite segment")
                
            if (len(tris_w_opp_segment) != 2):
                print("length of tris_w_opp_segment", len(tris_w_opp_segment))
                raise Exception("Invalid length")
            
            opp_tri_segments0 = tris_w_opp_segment[0].non_opposite_segments(opp_segment)
            opp_tri_segments1 = tris_w_opp_segment[1].non_opposite_segments(opp_segment)

            # find edges sharing similar endpoints from opp_segment
            left, right = [], []
            if (opp_tri_segments0[0].contains(opp_segment.p1)):
                left.append(opp_tri_segments0[0])
                right.append(opp_tri_segments0[1])
            else:
                left.append(opp_tri_segments0[1])
                right.append(opp_tri_segments0[0])
            
            if (opp_tri_segments1[0].contains(opp_segment.p1)):
                left.append(opp_tri_segments1[0])
                right.append(opp_tri_segments1[1])
            else:
                left.append(opp_tri_segments1[1])
                right.append(opp_tri_segments1[0])
            
            # remove both triangles in `tris_w_opp_segment` from `triangles`
            triangles.remove(tris_w_opp_segment[0])
            triangles.remove(tris_w_opp_segment[1])

            # dictionary update: remove `opp_segment` from dictionary and update values associated w/ segments from 
            # `opp_tri_segments0` and `opp_tri_segments1`
            
            for t1 in tris_w_opp_segment:
                segments_to_triangle_dict.get(t1.a).remove(t1)
                segments_to_triangle_dict.get(t1.b).remove(t1)
                segments_to_triangle_dict.get(t1.c).remove(t1)
            
            segments_to_triangle_dict.pop(opp_segment)

            # add two new triangles to triangles
            tri_left = triangle_from_segments(left)
            tri_right = triangle_from_segments(right)

            triangles.append(tri_left)
            triangles.append(tri_right)

            # dictionary update: update values associated with segments from `opp_tri_segments0` and `opp_tri_segments1`
            # and add in the new diagonal segment
            for t1 in [tri_left, tri_right]:
                update_dict(t1.a, t1, segments_to_triangle_dict)
                update_dict(t1.b, t1, segments_to_triangle_dict)
                update_dict(t1.c, t1, segments_to_triangle_dict)

            # update the stack using the two new triangles
            add_opposite_triangle(pt, tri_left, stack, segments_to_triangle_dict)
            add_opposite_triangle(pt, tri_right, stack, segments_to_triangle_dict)

def triangulate(points):
    '''
    Performs Delaunay Triangulation on a set of points

    points: a list of x,y coordinates

    returns a list of triangle objects
    '''

    st = super_triangle(points)
    triangles = [st]

    # maps edges to corresponding triangles (2 max)
    # note: useful for performing Lawson's search
    segments_to_triangle_dict = {st.a: [st], st.b: [st], st.c: [st]}

    for pt in points:
        insert_point(pt, triangles, segments_to_triangle_dict)

    return st, segments_to_triangle_dict, triangles

def remove_super_triangle(st, triangles):
    '''
    Removes the super triangle from a triangulation

    st: super triangle object
    triangle: a list of triangle objects

    returns a list of triangle objects
    '''

    final_triangles = []
    
    for triangle in triangles:
        if (triangle.contains(st.A_coords) or triangle.contains(st.B_coords) or triangle.contains(st.C_coords)):
            continue
        final_triangles.append(triangle)
    
    return final_triangles

############################## Arg parse function ##################################

def display_mosaic_baseline(path):
    '''
    Displays a Delaunay mosaic of the image

    path: path to image

    returns void
    '''
    image = cv2.imread(path)
    points = randomly_sample_points(image, 1000)

    st, _, triangles = triangulate(points)
    triangles = remove_super_triangle(st, triangles)
    fill_triangles(image, triangles)

    cv2.imshow("baseline", image)
    cv2.waitKey(0)


############################# TESTS ##################################

if __name__ == "__main__":
    original_img, blurred, output = get_edges("../../delaunay-mosaics/images/portraits/abe.jpeg", 5, 5, 25, 80)

    pts = sample_points_from_edges(output, 1200, 10)
    pts = np.concatenate((pts, randomly_sample_points(original_img, 300, 0.01, 30, mode="not corner")), axis=0)

    pts_tuples = []
    for pt in pts:
        tup = (int(pt[1]), int(pt[0]))
        if (tup not in pts_tuples):
            pts_tuples.append(tup)
    
    st, _, triangles = triangulate(pts_tuples)
    triangles = remove_super_triangle(st, triangles)
    
    fill_triangles(original_img, triangles)
    cv2.imshow("new image", original_img)
    cv2.waitKey(0)