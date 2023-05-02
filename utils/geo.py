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


############### Triangulation classes #######################
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
    
    def encloses(self, pt, ep=0.0005):
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
    
    def __eq__(self, other):
        pass
    
    def opposite_segment(self, pt):
        if (not self.a.contains(pt)):
            return self.a
        elif (not self.b.contains(pt)):
            return self.b
        elif (not self.c.contains(pt)):
            return self.c
        else:
            raise Exception("Input point is not a vertex of this triangle")
        
    def non_opposite_segments(self, opp_segment):
        segments = []
        if (self.a != opp_segment):
            segments.append(self.a)
        
        if (self.b != opp_segment):
            segments.append(self.b)

        if (self.c != opp_segment):
            segments.append(self.c)
        
        if (len(segments) >= 3):
            raise Exception("\"Opposite segment\" is not part of this triangle")
        
        return segments