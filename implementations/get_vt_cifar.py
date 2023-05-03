import random

import cv2
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from points_util import *

delaunay_color = (255,255,255)
points_color = (0, 0, 255)
point_cnt = 2000
K=1000
# 读入图片
img_path = "/data/haonan/dataset/archive/a/abbey/gsun_000bf27e490baf202e8a9c1a79ca6e41.jpg"
img = read_img(img_path)

img_orig = img.copy()

# 获取点
ps = create_point(point_cnt, img, mode="edge")

# 聚类
ps = run_cluster(ps,K)

rect = (0, 0, img.shape[1], img.shape[0])
subdiv = cv2.Subdiv2D(rect)

for p in ps:
    subdiv.insert(p)


# for p in ps:
#     draw_point(img, p, (0, 0, 255))
# cv2.imwrite('org_add_corner.png', img)


# print('Getting delaunay...')
# draw_delaunay(img, subdiv, (255, 255, 255))
# cv2.imwrite("delaunay.png", img)
#
# for p in ps:
#     draw_point(img, p, (0, 0, 255))
# cv2.imwrite('delaunay_add_ps.png', img)

print('Getting voronoi...')
# 为Voronoi 图分配空间
img_voronoi = np.zeros(img.shape, dtype = img.dtype)
# 绘制 Voronoi 图
voronoi_blur(img_orig, img_voronoi, subdiv)

cv2.imwrite('blur_voronoi.png', img_voronoi)