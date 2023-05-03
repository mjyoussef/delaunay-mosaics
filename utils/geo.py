import numpy as np

def dist(p1, p2):
    '''
    Computes the Euclidean distance between two points

    p1, p2: x,y coordinates of two points

    returns a real number
    '''
    return np.sqrt( ((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2) )

def orientation_of(p1, p2, p3, eps=0.0001):
    '''
    computes the orientation of three points

    p1, p2, p3: coordinates of points

    returns -1 for cw, 0 for colinear, and 1 for ccw
    '''
    det_matrix = np.array([[p1[0], p1[1], 1], [p2[0], p2[1], 1], [p3[0], p3[1], 1]])
    det_sign = np.linalg.det(det_matrix)

    if (det_sign <= eps and det_sign >= -eps):
        return 0
    elif (det_sign < 0):
        return -1
    else:
        return 1

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
    return np.sqrt(s * (s - a) * (s - b) * (s - c))


############### Triangulation classes #######################
class Segment():
    def __init__(self, p1, p2):
        pts = [p1, p2]
        list.sort(pts)

        self.p1 = pts[0]
        self.p2 = pts[1]

    def __hash__(self):
        str_rep = str(self.p1) + " " + str(self.p2)
        return hash(str_rep)
    
    def __eq__(self, other):
        return isinstance(other, Segment) and self.p1 == other.p1 and self.p2 == other.p2
    
    def midpoint(self):
        return ((self.p1[0] + self.p2[0]) / 2, (self.p1[1] + self.p2[1]) / 2)
    
    def contains(self, pt):
        return self.p1 == pt or self.p2 == pt
    
    def length(self):
        return dist(self.p1, self.p2)

class Triangle():
    def __init__(self, p1, p2, p3):

        self.line = False
        orientation = orientation_of(p1, p2, p3)
        if (orientation == 0): # colinear
            self.line = True
        
        self.a = Segment(p1, p2)
        self.b = Segment(p1, p3)
        self.c = Segment(p2, p3)

        self.A_coords = p3
        self.B_coords = p2
        self.C_coords = p1

        self.A, self.B, self.C, self.circumCenter = None, None, None, None

        if (not self.line):
            self.A = np.arccos(((self.b.length() ** 2) + (self.c.length() ** 2) - (self.a.length() ** 2)) / (2 * self.b.length() * self.c.length()))
            self.B = np.arccos(((self.a.length() ** 2) + (self.c.length() ** 2) - (self.b.length() ** 2)) / (2 * self.a.length() * self.c.length()))
            self.C = np.arccos(((self.a.length() ** 2) + (self.b.length() ** 2) - (self.c.length() ** 2)) / (2 * self.a.length() * self.b.length()))

            x = (self.A_coords[0] * np.sin(2*self.A)) + (self.B_coords[0] * np.sin(2*self.B)) + (self.C_coords[0] * np.sin(2*self.C))
            x /= (np.sin(2*self.A) + np.sin(2*self.B) + np.sin(2*self.C))

            y = (self.A_coords[1] * np.sin(2*self.A)) + (self.B_coords[1] * np.sin(2*self.B)) + (self.C_coords[1] * np.sin(2*self.C))
            y /= (np.sin(2*self.A) + np.sin(2*self.B) + np.sin(2*self.C))

            self.circumCenter = (x, y)
        else:
            max_segment = self.a
            if (self.b.length() > self.a.length()):
                max_segment = self.b

            if (self.c.length() > max_segment.length()):
                max_segment = self.c

            self.circumCenter = max_segment.midpoint()
            self.line_segment = max_segment
    
    def encloses(self, pt, eps=0.00005):
        if (self.line):
            return orientation_of(self.line_segment.p1, self.line_segment.p2, pt) == 0 \
            and pt[0] >= self.line_segment.p1[0] and pt[0] <= self.line_segment.p2[0]
        
        original_area = area_of(self.A_coords, self.B_coords, self.C_coords)
        a1 = area_of(pt, self.A_coords, self.B_coords)
        a2 = area_of(pt, self.A_coords, self.C_coords)
        a3 = area_of(pt, self.B_coords, self.C_coords)
        a_sum = a1 + a2 + a3

        return (original_area <= a_sum + eps) and (original_area >= a_sum - eps)
    
    def in_circumcircle(self, pt, eps=0.00000001, disp=False):
        pt_dist = dist(pt, self.circumCenter)
        radius = None
        if (self.line):
            radius = dist(self.line_segment.p1, self.circumCenter)
        else:
            radius = dist(self.A_coords, self.circumCenter)

        # more likely to consider a point as not being in the circumcircle
        if (disp):
            print(pt_dist, radius)
        return pt_dist < radius - eps
    
    def __eq__(self, other):
        if (not isinstance(other, Triangle)):
            return False
        
        coords = [self.A_coords, self.B_coords, self.C_coords]
        return coords.count(other.A_coords) > 0 and coords.count(other.B_coords) > 0 and coords.count(other.C_coords) > 0
    
    def __hash__(self):
        coords = [self.A_coords, self.B_coords, self.C_coords]
        list.sort(coords)
        str_rep = str(coords[0]) + " " + str(coords[1]) + " " + str(coords[2])
        return hash(str_rep)
    
    def opposite_segment(self, pt):
        if (not self.a.contains(pt)):
            return self.a
        elif (not self.b.contains(pt)):
            return self.b
        elif (not self.c.contains(pt)):
            return self.c
        else:
            raise Exception("Invalid triangle: every segment has a common point")
        
    def non_opposite_segments(self, opp_segment):
        segments = []
        if (self.a != opp_segment):
            segments.append(self.a)
        
        if (self.b != opp_segment):
            segments.append(self.b)

        if (self.c != opp_segment):
            segments.append(self.c)
        
        if (len(segments) > 2):
            raise Exception("\"Opposite segment\" is not part of this triangle")
        
        return segments
    
    def contains(self, pt):
        return self.A_coords == pt or self.B_coords == pt or self.C_coords == pt
    
    def contains_segment(self, segment):
        return self.a == segment or self.b == segment or self.c == segment