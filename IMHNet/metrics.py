import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, jaccard_score
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff
import cv2
from scipy.ndimage import zoom

from scipy.ndimage import binary_erosion, generate_binary_structure
import GeodisTK
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if (dim == 2):
        strt = generate_binary_structure(2, 1)
    else:
        strt = generate_binary_structure(3, 1)  
    ero = binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


def binary_hausdorff95(s, g, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    import sys
    sys.path.append(r'./.venv/Lib/site-packages')

    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if (image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim == 3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    if dist1 == 10000000000.0:
        dist2 = 49.0
        dist1 = 50.0
    else:
        dist_list2 = g_dis[s_edge > 0]
        dist_list2 = sorted(dist_list2)
        dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)