import cv2
import numpy as np


def create_point(cnt, img, mode="corner"):
    print('Getting points...')
    # ps = []
    if mode=="corner":
        # 对特征点进行角点检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, cnt, 0.01, 2)
        corners = np.squeeze(corners)
        return corners
    elif mode=="edge":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # Get the coordinates of the points on the edges
        coords = np.column_stack(np.where(edges > 0))

        # Randomly sample 100 points from the coordinates
        n_samples = 100
        samples = coords[np.random.choice(coords.shape[0], n_samples, replace=False), :]
        return samples

    else:
        x = np.random.randint(1, img.shape[0]-1, (cnt, 1))
        y = np.random.randint(1, img.shape[1]-1, (cnt, 1))
        return np.concatenate([x,y], axis=1)