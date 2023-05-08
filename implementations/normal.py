import sys
sys.path.append("../../delaunay-mosaics")

import cv2
import numpy as np
from utils.geo import Triangle
from utils.sampling import randomly_sample_points, sample_points_from_edges, get_edges, add_pts_to_pts
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
    Updates the segments-to-triangle dictionary with a segment and its corresponding triangle

    seg: a segment object
    triangle: a triangle object
    segments_to_triangle_dict: segments dictionary

    returns void
    '''
    if (segments_to_triangle_dict.get(seg) == None):
        segments_to_triangle_dict.update({seg: set([triangle])})
    else:
        segments_to_triangle_dict.get(seg).add(triangle)

def update_dict_from_triangles(triangles, segments_to_triangle_dict, remove=True):
    '''
    Removes (or adds) triangles from the segments to triangle dictionary

    triangles: a list of triangle objects
    segments_to_triangle_dict: dictionary

    returns void
    '''
    for t in triangles:
        for seg in [t.a, t.b, t.c]:
            if (remove):
                segments_to_triangle_dict.get(seg).remove(t)
            else:
                update_dict(seg, t, segments_to_triangle_dict)

def swap_diagonal(segment_to_triangle_dict, old_diagonal):
    left, right = [], []
    tris_w_diag = segment_to_triangle_dict.get(old_diagonal).copy()

    for t in tris_w_diag:
        non_opp_segments = t.non_opposite_segments(old_diagonal)
        if (non_opp_segments[0].contains(old_diagonal.p1)):
            left.append(non_opp_segments[0])
            right.append(non_opp_segments[1])
        else:
            left.append(non_opp_segments[1])
            right.append(non_opp_segments[0])
    
    left_tri = triangle_from_segments(left)
    right_tri = triangle_from_segments(right)

    # dictionary update
    update_dict_from_triangles(tris_w_diag, segment_to_triangle_dict, remove=True)
    segment_to_triangle_dict.pop(old_diagonal)

    update_dict_from_triangles([left_tri, right_tri], segment_to_triangle_dict, remove=False)
    
    # return reference to the new diagonal
    return left_tri.opposite_segment(old_diagonal.p1)

def insert_point(pt, triangles, segments_to_triangle_dict):
    '''
    Subroutine for Lawson's search algorithm, which updates the
    triangulation for a given input point

    pt: x,y coordinates
    triangles: a set of triangle objects
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
    update_dict_from_triangles([init_t], segments_to_triangle_dict, remove=True)
    triangles.remove(init_t)

    for t in [new_tri1, new_tri2, new_tri3]:
        # add opposite triangles to stack
        add_opposite_triangle(pt, t, stack, segments_to_triangle_dict)

        # add triangle to triangles
        triangles.add(t)

    # add new line segments + references to new triangle in the dictionary
    update_dict_from_triangles([new_tri1, new_tri2, new_tri3], segments_to_triangle_dict, remove=False)
    
    while (len(stack) > 0): # see Lawson's algorithm

        # triangle opposite to `pt`
        t, opp_segment = stack.pop()

        if (t.in_circumcircle(pt)): # swap diagonals of the convex quadrilateral

            tris_w_opp_segment = segments_to_triangle_dict.get(opp_segment)

            # remove both triangles in `tris_w_opp_segment` from `triangles`
            for t1 in tris_w_opp_segment:
                triangles.remove(t1)

            # create two new triangles by swapping the diagonal
            new_diag = swap_diagonal(segments_to_triangle_dict, opp_segment)

            new_tris = segments_to_triangle_dict.get(new_diag)
            for t1 in new_tris:
                triangles.add(t1)

                # update the stack
                add_opposite_triangle(pt, t1, stack, segments_to_triangle_dict)

def triangulate(points):
    '''
    Performs Delaunay Triangulation on a set of points

    points: a list of x,y coordinates

    returns a list of triangle objects
    '''

    st = super_triangle(points)
    triangles = set([st])

    # maps edges to corresponding triangles
    segments_to_triangle_dict = {st.a: set([st]), st.b: set([st]), st.c: set([st])}

    for pt in points:
        insert_point(pt, triangles, segments_to_triangle_dict)

    return st, segments_to_triangle_dict, triangles

def remove_super_triangle(st, triangles):
    '''
    Removes any triangle containing a vertex from the super triangle

    st: super triangle object
    triangle: a set of triangle objects

    returns a list of triangle objects
    '''

    final_triangles = []
    
    for triangle in triangles:
        if (triangle.contains(st.A_coords) or triangle.contains(st.B_coords) or triangle.contains(st.C_coords)):
            continue
        final_triangles.append(triangle)
    
    return final_triangles

############################## Arg parse functions ##################################

def display_mosaic_baseline(path, args_dict):
    '''
    Displays a Delaunay mosaic of the image

    path: path to image
    args_dict: argument dictionary

    returns void
    '''
    image = cv2.imread(path)
    points = randomly_sample_points(image, args_dict.get("num_points"), 0, args_dict.get("min_dist"), mode='not corner')

    st, _, triangles = triangulate(points)
    triangles = remove_super_triangle(st, triangles)
    print(len(triangles))
    fill_triangles(image, triangles)

    cv2.imshow("baseline", image)
    cv2.waitKey(0)

def display_mosaic_w_edges(path, args_dict):
    '''
    Displays a Delaunay mosaic of the image w/ consideration for the edges

    path: path to image
    args_dict: argument dictionary

    returns void
    '''

    print("Running edge detection...\n")
    original_img, _, output = get_edges(path, args_dict.get("sigma_x"), args_dict.get("sigma_y"), 20, 80)
    
    print("Collecting points from edges...")
    pts = sample_points_from_edges(output, args_dict.get("num_edge_points"), args_dict.get("min_dist"))
    print(f"Found {len(pts)} points from edges\n")
    print("Collecting additional points")
    additional_pts = add_pts_to_pts(original_img, pts, args_dict.get("num_add_points"), args_dict.get("min_dist"))
    print(f"Found {len(additional_pts)} additional points\n")

    print("Triangulating...")
    st, _, triangles = triangulate(pts + additional_pts)
    triangles = remove_super_triangle(st, triangles)
    print(f"Generated {len(triangles)} triangles\n")

    fill_triangles(original_img, triangles)
    cv2.imshow("Triangulated Image", original_img)
    cv2.waitKey(0)