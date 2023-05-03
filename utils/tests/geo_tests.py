import sys
sys.path.append("../../../delaunay-mosaics")

from utils.geo import *
import unittest
import numpy as np

class OrientationTests(unittest.TestCase):
    def test_colinear(self):
        p1, p2, p3 = (0,0), (1,1), (8,8)
        self.assertEqual(orientation_of(p1, p2, p3), 0, "colinear test")
        self.assertEqual(orientation_of(p2, p3, p1), 0, "colinear test -- shuffled 1")
        self.assertEqual(orientation_of(p3, p2, p1), 0, "colinear test -- shuffled 2")
        
    def dummy_colinear(self):
        p1, p2, p3 = (0,0), (2,2), (40,30.9)
        self.assertEqual(orientation_of(p1, p2, p3), -1, "dummy colinear test -- cw")

        p3 = (40, 40.1)
        self.assertEqual(orientation_of(p1, p2, p3), 1, "dummy colinear test -- ccw")
    
    def test_cw(self):
        p1, p2, p3 = (-1, -1), (2, 2), (3, -4)
        self.assertEqual(orientation_of(p1, p2, p3), -1, "cw test")
    
    def test_ccw(self):
        p1, p2, p3 = (-3, -8), (0, 1), (2, 30.8)
        self.assertEqual(orientation_of(p1, p2, p3), 1, "ccw test")

class IntersectionTests(unittest.TestCase):
    def test_no_intersection(self):
        e1, e2 = ((0,0), (5,5)), ((1,0), (5,0))
        self.assertFalse(segments_intersect(e1, e2), "obvious no intersection")
    
    def test_matching_eps(self):
        e1, e2 = ((-1,3), (3,7)), ((-1,3), (8,0))
        self.assertFalse(segments_intersect(e1, e2), "matching endpoints")

    def test_obvious_intersection(self):
        e1, e2 = ((-1,3), (3,7)), ((-5,6), (8,6))
        self.assertTrue(segments_intersect(e1, e2), "obvious intersection")
    
    def test_intersection_at_ep(self):
        e1, e2 = ((-1,3), (3,7)), ((-2, -3), (0, 9))
        self.assertTrue(segments_intersect(e1, e2), "intersection at ep")
    
    def test_colinear_no_intersection(self):
        e1, e2 = ((0,0), (2,2)), ((2.5,2.5), (7,7))
        self.assertFalse(segments_intersect(e1, e2), "colinear w/ no intersection")
    
    def test_colinear_intersection(self):
        e1, e2 = ((0,0), (2,2)), ((1,1), (3,3))
        self.assertTrue(segments_intersect(e1, e2), "colinear w/ intersection")

class AreaTests(unittest.TestCase):
    def test_colinear(self):
        p1, p2, p3 = (0,0), (1,1), (2,2)
        self.assertAlmostEqual(area_of(p1, p2, p3), 0, "colinear test")
    
    def test_triangle(self):
        p1, p2, p3 = (-2,3), (-3,-1), (3,-2)
        self.assertAlmostEqual(area_of(p1, p2, p3), 12.5, "simple triangle")

class SegmentTests(unittest.TestCase):
    def test_same_segments(self):
        seg1, seg2 = Segment((1,1), (0,0)), Segment((0,0), (1,1))
        seg3, seg4 = Segment((-1,-1), (-1,4)), Segment((-1,4), (-1,-1))

        self.assertEqual(seg1, seg2, "same segments 1")
        self.assertEqual(seg3, seg4, "same segments 2")
    
    def test_diff_segments(self):
        seg1, seg2 = Segment((1,1), (3,4)), Segment((3,3), (1,4))
        self.assertNotEqual(seg1, seg2, "different segments")

    def test_midpoint(self):
        seg = Segment((2,2), (3,3))
        self.assertTupleEqual(seg.midpoint(), (2.5,2.5), "midpoint test")

class TriangleTests(unittest.TestCase):
    def test_angles(self):
        t = Triangle((0,0), (3,4), (30,1))
        r1, r2, r3 = t.a.length() / np.sin(t.A), t.b.length() / np.sin(t.B), t.c.length() / np.sin(t.C)

        self.assertAlmostEqual(r1, r2)
        self.assertAlmostEqual(r2, r3)

        self.assertAlmostEqual(t.A + t.B + t.C, np.pi)
    
    def test_circumcenter(self):
        t_acute = Triangle((-3.5, 8), (3,4), (30,1))
        cc = t_acute.circumCenter
        self.assertAlmostEqual(cc[0], 20.665, places=3)
        self.assertAlmostEqual(cc[1], 39.987, places=3)

        t_average = Triangle((2.4, 12), (2.4, 0), (5, 6))
        cc = t_average.circumCenter
        self.assertAlmostEqual(cc[0], -3.223, places=3)
        self.assertAlmostEqual(cc[1], 6, places=3)

        t_colinear = Triangle((0,0), (1,1), (4,4))
        cc = t_colinear.circumCenter
        self.assertAlmostEqual(cc[0], 2)
        self.assertAlmostEqual(cc[0], 2)
        self.assertFalse(t_colinear.in_circumcircle((4.9, 2)))
    
    def test_encloses(self):
        t = Triangle((0,0), (0,8), (7,8))
        self.assertTrue(t.encloses((0,4)))
        self.assertTrue(t.encloses((1,5)))
        self.assertFalse(t.encloses((0,9)))
    
    def test_equality(self):
        t1 = Triangle((-3.5,8), (3,4), (30,1))
        t2 = Triangle((3,4), (30,1), (-3.5,8))
        self.assertEqual(t1, t2)

        t1 = Triangle((-3.5,8), (3,4), (30,1))
        t2 = Triangle((3,4), (30,1), (-3.4,8))
        self.assertNotEqual(t1, t2)
    
    def test_opposite_segments_utils(self):
        t = Triangle((0,0), (3,3), (8,20))
        opp_segment = Segment((0,0), (8,20))
        self.assertEqual(t.opposite_segment((3,3)), opp_segment)

        non_opp_segments = t.non_opposite_segments(opp_segment)
        self.assertEqual(non_opp_segments, [Segment((3,3), (0,0)), Segment((3,3), (8,20))])

if __name__ == '__main__':
    unittest.main()