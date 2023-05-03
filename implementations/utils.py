import cv2
import numpy as np
import random

def read_img(img_path):
    img = cv2.imread(img_path)
    cv2.imwrite("org.png", img)
    return img


def run_cluster(ps, K):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(ps, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    res = []
    for c in center:
        res.append((int(c[0]), int(c[1])))
    return res



# 绘制 delaunay 三角剖分
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


def draw_point(img, p, color):
    cv2.circle(img, p, 1, color, cv2.FILLED, cv2.LINE_AA, 0)


def draw_voronoi(img, subdiv):
    (facets, centers) = subdiv.getVoronoiFacetList([])
    for i in range(0, len(facets)):
        # ifacet_arr = []
        # for f in facets:
        #     for f_i in f:
        #         ifacet_arr.append(f_i)
        # ifacet = np.concatenate([ifacet_arr]).astype(int)
        # ifacet = np.array(ifacet_arr, int)
        ifacet = np.array(facets[i],int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # color = (0,255,0)
        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (int(centers[i][0]), int(centers[i][1])), 3, (0, 0, 0), cv2.FILLED, cv2.LINE_AA, 0)


def draw_voronoi_wo_blur(org_img, img_blur, subdiv):
    (facets, centers) = subdiv.getVoronoiFacetList([])
    for i in range(0, len(facets)):
        ifacet = np.array(facets[i], int)
        ifacets = np.array([ifacet])
        cv2.polylines(org_img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)


def voronoi_blur(org_img, img_blur, subdiv):
    (facets, centers) = subdiv.getVoronoiFacetList([])
    for i in range(0, len(facets)):
        ifacet = np.array(facets[i], int)
        ifacet_clip = np.clip(ifacet, 0, 255)
        color = tuple(np.mean(org_img[ifacet_clip[:, 1], ifacet_clip[:, 0]], axis=0).astype(int))
        color = tuple(int(i) for i in color)
        cv2.fillConvexPoly(img_blur, ifacet, color, cv2.LINE_AA, 0)







