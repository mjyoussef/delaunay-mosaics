# Delaunay Mosaics: Generated Edge-Informed Triangulations of Images

## Baseline
The baseline approach randomly samples points in an image, creates a triangulation from them, and fills in the color of each cell using the average pixel color. 

This approach is decent but can sometimes produce irregular mosaics, especially if the number of points sampled is too small for a highly detailed image. 

## Approach 1: Edge Detector + Random Point Sampling
We can modify the baseline approach by outlining edges in an image (ie. using the Canny Edge Detector), randomly sampling points along the edges, and adding any additional points. 

## Aproach 2: Edge Detector + Random Point Sampling + Constrained Delaunay Triangulation
The goal of approach 1 is to make sure that cells edges in an image also seperate cells in the triangulation. This cannot be guarenteed, especially when the density of the points is high, because some edges in the triangulation may end up crossing edges in the image.

To remedy this, we use constrained Delaunay Triangulation, which is a type of Delaunay Triangulation that incorporates a specific set of edges. By sampling *edges* instead of points along the edges in an image and passing them into a constrained Delaunay Triangulation algorithm, we can ensure that only few edges in the triangulation cross edges in the image.
