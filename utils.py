import cv2


def get_edges(path, sigmaX, sigmaY, thresh1, thresh2):
    '''
    Extracts edges from images

    sigmaX: x variance for Gaussian filter
    sigmaY: y variance for Gaussian filter
    thresh1: lower threshold for hysteris
    thresh2: upper threshold for hysteris

    returns blurred output and output w/ outlined edges
    '''
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (sigmaX, sigmaY), 0)
    output = cv2.Canny(blurred, thresh1, thresh2)

    # cv2.imshow("blurred", blurred)
    # cv2.imshow("output", output)
    # cv2.waitKey(0)

    return blurred, output

def sample_points_from_edges(img, num_points):
    '''
    Randomly samples points along the edges of an image

    img: image that has already been run through an edge detector
    num_points: number of points to sample

    returns points in the image as a list of tuples
    '''
    
    pass


def sample_edges_from_edges(img, min_length, max_length):
    '''
    Randomly samples edges that are bounded by certain length threshold

    img: image that has already been run through an edge detector
    min_length: minimum edge length (L2 distance)
    max_length: maximum edge length (L2 distance)

    returns edges in the image as a list of pairs of tuples
    '''

    pass

if __name__ == '__main__':
    sample_points_from_edges("images/portraits/abe.jpeg")