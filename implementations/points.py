from utils.geo import orientation_of, area_of, dist
import numpy as np

class Segment():
    def __init__(self, p1, p2):
        pts = [p1, p2]
        list.sort(pts)

        self.p1 = pts[0]
        self.p2 = pts[1]

    def __hash__(self):
        return str(self.p1) + " " + str(self.p2)
    
    def __eq__(self, other):
        return isinstance(other, Segment) and self.p1 == other.p1 and self.p2 == other.p2
    
    def midpoint(self):
        return ((self.p1[0] + self.p2[0]) / 2, (self.p1[1] + self.p2[1]) / 2)
    
    def contains(self, pt):
        return self.p1 == pt or self.p2 == pt


class Triangle():
    def __init__(self, p1, p2, p3):

        self.line = False
        if (orientation_of(p1, p2, p3) == 0): # colinear
            self.line = True
        
        self.a = Segment(p1, p2)
        self.b = Segment(p1, p3)
        self.c = Segment(p2, p3)

        self.A_coords = p3
        self.B_coords = p2
        self.C_coords = p1

        self.A, self.B, self.C, self.circumCenter = None, None, None, None

        if (not self.line):
            self.A = np.arccos(((self.b ** 2) + (self.c ** 2) - (self.a ** 2)) / (2 * self.b * self.c))
            self.B = np.arccos(((self.a ** 2) + (self.c ** 2) - (self.b ** 2)) / (2 * self.a * self.c))
            self.C = np.arccos(((self.a ** 2) + (self.b ** 2) - (self.c ** 2)) / (2 * self.a * self.b))

            x = (self.A_coords[0] * np.sin(2*self.A)) + (self.B_coords[0] * np.sin(2*self.B)) + (self.C_coords[0] * np.sin(2*self.C))
            x /= (np.sin(2*self.A) + np.sin(2*self.B) + np.sin(2*self.C))

            y = (self.A_coords[1] * np.sin(2*self.A)) + (self.B_coords[1] * np.sin(2*self.B)) + (self.C_coords[1] * np.sin(2*self.C))
            y /= (np.sin(2*self.A) + np.sin(2*self.B) + np.sin(2*self.C))

            self.circumCenter = (x, y)
    
    def is_enclosed(self, pt, ep=0.0005):
        if (self.line):
            raise Exception("Cannot check if a point is enclosed by a line")
        
        original_area = area_of(self.A_coords, self.B_coords, self.C_coords)
        a1 = area_of(pt, self.A_coords, self.B_coords)
        a2 = area_of(pt, self.A_coords, self.C_coords)
        a3 = area_of(pt, self.B_coords, self.C_coords)
        a_sum = a1 + a2 + a3

        return (original_area <= a_sum + ep) and (original_area >= a_sum - ep)
    
    def in_circumcircle(self, pt, ep=0.001):
        pt_dist = dist(pt, self.circumCenter)
        radius = dist(self.A_coords, self.circumCenter)

        return pt_dist <= radius + ep