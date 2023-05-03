import cv2
from baseline import triangulate, remove_super_triangle, triangle_from_segments, update_dict
from utils.geo import segments_intersect
from collections import deque

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
    tris_w_seg = seg_dict.get(seg)

    for t in tris_w_seg:
        for t2 in tris_w_seg:
            for pt in [t.A_coords, t.B_coords, t.C_coords]:
                if (t2.in_circumcircle(pt)):
                    return False

    return True

def restore_triangulation(seg_dict, new_segments, constrained_segment):
    swaps = None
    while (swaps == None or swaps > 0):
        new_segments2 = []
        for seg in new_segments:
            if (seg != constrained_segment):
                if (not check_delaunay(seg_dict, seg)):
                    new_diag = swap_diagonal(seg_dict, seg)
                    new_segments2.append(new_diag)
                    continue
            new_segments2.append(seg)
        
        new_segments = new_segments2

def get_contour(triangles):
    pts = []
    for t in triangles:
        if (t.A_coords not in pts):
            pts.append(t.A_coords)
        if (t.B_coords not in pts):
            pts.append(t.B_coords)
        if (t.C_coords not in pts):
            pts.append(t.C_coords)
    
    return pts

def swap_diagonal(seg_dict, old_diagonal):
    left, right = [], []
    tris_w_diag = seg_dict.get(old_diagonal).copy()

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
    for t in tris_w_diag:
        for seg in [t.a, t.b, t.c]:
            seg_dict.get(seg).remove(t)
    
    seg_dict.pop(old_diagonal)

    for t in [left_tri, right_tri]:
        for seg in [t.a, t.b, t.c]:
            update_dict(t, seg_dict)
    
    # return reference to the new diagonal
    return left_tri.opposite_segment(old_diagonal.p1)


def remove_intersecting_edges(seg_dict, constrained_segment):

    new_segments = []

    # find segments that intersect the constrained segment
    intersecting_segments = deque([])
    for key in seg_dict.keys():
        if (segments_intersect(key, constrained_segment)):
            intersecting_segments.append(key)

    while (len(intersecting_segments) > 0):

        seg = intersecting_segments.popleft()
        contour = get_contour(seg_dict.get(seg))

        # Check if the triangles containing the segment form a convex quadrilateral
        # if no reappend to the queue
        # if yes, swap diagonals

        if (not cv2.isContourConvex(contour)):
            intersecting_segments.append(seg)
        else:
            new_diag = swap_diagonal(seg_dict, seg)
            if (segments_intersect(new_diag, constrained_segment)):
                intersecting_segments.append(new_diag)
            else:
                new_segments.append(new_diag)
    
    return new_segments

def constrained_triangulation(pts, segments):
    for seg in segments:
        if (seg.p1 not in pts):
            pts.append(seg.p1)
        if (seg.p2 not in pts):
            pts.append(seg.p2)
        
    st, seg_dict, _ = triangulate(pts)

    for seg in segments:
        if (seg_dict.get(seg) == None):
            new_segments = remove_intersecting_edges(seg_dict, seg)
            restore_triangulation(seg_dict, new_segments, seg)
    
    # collect all the triangles from the dictionary and return
    triangles = set()
    for key in seg_dict.keys():
        for t in seg_dict.get(key):
            triangles.add(t)
    
    return list(triangles)