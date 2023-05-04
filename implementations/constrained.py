import cv2
from baseline import triangulate, remove_super_triangle, triangle_from_segments, update_dict, fill_triangles
from utils.geo import segments_intersect
from collections import deque
from utils.geo import orientation_of, dist, Segment, Triangle
from utils.sampling import sample_edges_from_edges, randomly_sample_points, get_edges
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
    tris_w_seg = seg_dict.get(seg)

    for t in tris_w_seg:
        for t2 in tris_w_seg:
            for pt in [t.A_coords, t.B_coords, t.C_coords]:
                if (t2.in_circumcircle(pt)):
                    return False

    return True

def restore_triangulation(seg_dict, new_segments, constrained_segment):
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
    pts = []
    if (len(triangles) != 2):
        print("num of triangles", len(triangles))
        raise Exception("Invalid number of triangles in dictionary")
    
    eps = []
    for t in triangles:
        for coord in [t.A_coords, t.B_coords, t.C_coords]:
            if (not diag.contains(coord)):
                eps.append(coord)
    
    return [eps[0], diag.p1, eps[1], diag.p2]

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
            update_dict(seg, t, seg_dict)
    
    # return reference to the new diagonal
    return left_tri.opposite_segment(old_diagonal.p1)


def remove_intersecting_edges(seg_dict, constrained_segment):

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


###################### Tests ####################

if __name__ == '__main__':
    dist_thresh = 30
    theta_thresh = np.pi/6

    # load img + run edge detection
    original_img, blurred, output = get_edges("../../delaunay-mosaics/images/portraits/abe.jpeg", 5, 5, 25, 80)

    # get constrained edges + pts
    edges = sample_edges_from_edges(output, 15, dist_thresh, theta_thresh, radius=1)
    edges_transpose = []
    for e in edges:
        edges_transpose.append(Segment((e[0][1], e[0][0]), (e[1][1], e[1][0])))
    edges = edges_transpose

    pts = randomly_sample_points(original_img, 300, 0.01, dist_thresh-10, mode="not corner")

    # filter points
    filtered_pts = []
    for pt in pts:
        isValidPt = True
        for e in edges:
            if (orientation_of(e.p1, e.p2, pt) == 0 and pt[0] >= e.p1[0] and pt[0] <= e.p2[0]):
                isValidPt = False
                break
            
            if (dist(e.p1, pt) < dist_thresh-10 or dist(e.p2, pt) < dist_thresh-10):
                isValidPt = False
                break
        
        if (isValidPt):
            filtered_pts.append((pt[1], pt[0]))
    
    st, seg_dict, triangles = constrained_triangulation(filtered_pts, edges)
    final_triangles = remove_super_triangle(st, triangles)
    print(len(triangles))
    fill_triangles(original_img, final_triangles)
    cv2.imshow("new image", original_img)
    cv2.waitKey(0)