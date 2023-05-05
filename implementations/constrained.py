import cv2
from normal import triangulate, remove_super_triangle, swap_diagonal, fill_triangles
from collections import deque
from utils.geo import segments_intersect
from utils.sampling import sample_edges_from_edges, get_edges, add_pts_to_segments
import numpy as np

'''
Edge-constrained Delaunay triangulation...

Step 1: Run Canny on the image

Step 2: Randomly sample edges from the Canny output

Step 3: Randomly sample points in the image

Step 4: Input the edges (from step 2) and the points (step 3) into a 
Constrained Delaunay Triangulation

Step 5: Update the color of every pixel in a triangle to the average color in that triangle

Step 6: Save and display the mosaic from step 5

'''

def check_delaunay(seg_dict, seg):
    '''
    Checks if the triangles containing seg maintain the Delaunay property

    seg_dict: segment to triangles dictionary
    seg: segment

    return true/false
    '''
    tris_w_seg = seg_dict.get(seg)

    for t in tris_w_seg:
        for t2 in tris_w_seg:
            for pt in [t.A_coords, t.B_coords, t.C_coords]:
                if (t2.in_circumcircle(pt)):
                    return False

    return True

def restore_triangulation(seg_dict, new_segments, constrained_segment):
    '''
    Swapping procedure that restores the Delaunay property

    seg_dict: segment to triangles dictionary
    new_segments: newly created segments that don't intersect `constrained_segment`
    constrained_segment: the constrained segment

    returns void (updates segment dictionary)
    '''
    swaps = 1
    while (swaps > 0):
        swaps = 0
        new_segments2 = []
        for seg in new_segments:
            if (seg != constrained_segment):
                if (not check_delaunay(seg_dict, seg)):
                    new_diag = swap_diagonal(seg_dict, seg)
                    new_segments2.append(new_diag)
                    swaps += 1
                    continue
            new_segments2.append(seg)
        
        new_segments = new_segments2

def get_contour(triangles, diag):
    '''
    Creates a quadrilateral from the triangles

    triangles: a list of triangles
    diag: diagonal segment

    returns a list of points
    '''
    if (len(triangles) != 2):
        raise Exception("Invalid number of triangles in dictionary")
    
    eps = []
    for t in triangles:
        for coord in [t.A_coords, t.B_coords, t.C_coords]:
            if (not diag.contains(coord)):
                eps.append(coord)
    
    return [eps[0], diag.p1, eps[1], diag.p2]


def remove_intersecting_edges(seg_dict, constrained_segment):
    '''
    Removes any segments that intersect the constrained segment

    seg_dict: segment to triangles dictionary
    constrained_segment: a constrained segment

    returns a list of newly created segments
    '''

    new_segments = []

    # find segments that intersect the constrained segment
    intersecting_segments = deque([])
    for key in seg_dict.keys():
        if (segments_intersect((key.p1, key.p2), (constrained_segment.p1, constrained_segment.p2))):
            intersecting_segments.append(key)

    while (len(intersecting_segments) > 0):

        seg = intersecting_segments.popleft()
        contour = get_contour(seg_dict.get(seg), seg)

        # Check if the triangles containing the segment form a convex quadrilateral
        # if no reappend to the queue
        # if yes, swap diagonals

        if (not cv2.isContourConvex(np.array(contour))):
            intersecting_segments.append(seg)
        else:
            new_diag = swap_diagonal(seg_dict, seg)
            if (segments_intersect((new_diag.p1, new_diag.p2), (constrained_segment.p1, constrained_segment.p2))):
                intersecting_segments.append(new_diag)
            else:
                new_segments.append(new_diag)
    
    return new_segments

def constrained_triangulation(pts, segments):
    '''
    Runs constrained Delaunay triangulation on a list of points and segments

    pts: a list of x,y coordinates
    segments: a list of required segments

    returns a super triangle, segment dictionary, and list of triangles
    '''
    for seg in segments:
        if (seg.p1 not in pts):
            pts.append(seg.p1)
        if (seg.p2 not in pts):
            pts.append(seg.p2)
        
    st, seg_dict, _ = triangulate(pts)

    # for each constrained segment not already in the dictionary...
    for seg in segments:
        if (seg_dict.get(seg) == None):
            new_segments = remove_intersecting_edges(seg_dict, seg)
            restore_triangulation(seg_dict, new_segments, seg)
    
    # collect all the triangles from the dictionary and return
    triangles = set()
    for key in seg_dict.keys():
        for t in seg_dict.get(key):
            triangles.add(t)
    
    return st, seg_dict, list(triangles)

############################ Arg parse functions #############################

def display_mosaic_w_constrained_edges(path, args_dict):
    '''
    Triangulates an image using sampled edges as constrained segments

    path: path to the image
    args_dict: argument dictionary

    returns void
    '''

    print("Running edge detection...\n")
    original_img, _, output = get_edges(path, args_dict.get("sigma_x"), args_dict.get("sigma_y"), 20, 80)
    
    print("Collecting sample edges...")
    segments = sample_edges_from_edges(output, args_dict.get("min_length"), args_dict.get("min_dist"), args_dict.get("theta_thresh"))
    print(f"Found {len(segments)} segments\n")

    print("Collecting additional points...")
    additional_pts = add_pts_to_segments(original_img, segments, args_dict.get("num_add_pts"), args_dict.get("min_dist"))
    print(f"Found {len(additional_pts)} additional points\n")

    print("Triangulating...")
    st, _, triangles = constrained_triangulation(segments, additional_pts)
    triangles = remove_super_triangle(st, triangles)
    print(f"Generated {len(triangles)} triangles\n")

    fill_triangles(original_img, triangles)
    cv2.imshow("Triangulated Image", original_img)
    cv2.waitKey(0)