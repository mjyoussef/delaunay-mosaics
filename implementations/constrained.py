import cv2
from baseline import triangulate, remove_super_triangle

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

