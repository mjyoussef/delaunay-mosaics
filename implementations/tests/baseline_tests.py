import sys
sys.path.append("../../../delaunay-mosaics")

from implementations.baseline import super_triangle, triangulate, remove_super_triangle
from utils.geo import segments_intersect
import unittest

import numpy as np


class SuperTriangleTests(unittest.TestCase):
    def test_one_pt(self):
        pts = [(5,8)]
        st = super_triangle(pts)
        for pt in pts:
            self.assertTrue(st.encloses(pt))

    def test_colinear_pts(self):
        pts = [(2,2), (5,5), (30,30)]
        st = super_triangle(pts)
        for pt in pts:
            self.assertTrue(st.encloses(pt))

    def test_multiple_pts(self):
        pts = [(-1,5), (-1,20), (100,5), (100,20), (-3,-2), (45, 18)]
        st = super_triangle(pts)
        for pt in pts:
            self.assertTrue(st.encloses(pt))

class TriangulateTests(unittest.TestCase):
    def test_triangulate(self):
        pts = [(-1,5), (-1,90), (100,5), (100,90)]

        # check that no triangles intersect
        st, seg_dict, triangles = triangulate(pts)
        for t1 in triangles:
            segs1 = [t1.a, t1.b, t1.c]
            for t2 in triangles:
                segs2 = [t2.a, t2.b, t2.c]

                for seg1 in segs1:
                    for seg2 in segs2:
                        self.assertFalse(segments_intersect((seg1.p1, seg1.p2), (seg2.p1, seg2.p2)))
        
        # check to make sure triangles are adjacent
        keys = seg_dict.keys()
        for key in keys:
            if (st.contains_segment(key)):
                self.assertEqual(len(seg_dict.get(key)), 1)
            else:
                self.assertEqual(len(seg_dict.get(key)), 2)

        # check that no point is inside the circumcircle of another triangle
        pts_w_st = pts.copy()
        for pt in [st.A_coords, st.B_coords, st.C_coords]:
            if (not pt in pts_w_st):
                pts_w_st.append(pt)
        
        new_triangles = remove_super_triangle(st, triangles)
        for t in new_triangles:
            print(t.A_coords, t.B_coords, t.C_coords)
        
        for pt in pts_w_st:
            for t in triangles:
                if (t.in_circumcircle(pt, disp=False)):
                    print(pt)
                    print(t.A_coords, t.B_coords, t.C_coords)


if __name__ == '__main__':
    unittest.main()