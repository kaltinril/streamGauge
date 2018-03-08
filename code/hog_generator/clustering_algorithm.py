import numpy as np
import skfuzzy.cluster as fuzzy
import hog_generator as hg
import cv2
# import matplotlib.pyplot as plt

def fuzzy_c_means(num_categories):
    data, metadata = hg.load_hogs('./HOG Files')
    data = hg.PCA(data, 36)
    # data must be of size (S, N) where S is the number of features in a vector, and N is the number of feature vectors
    data = data.T
    centroids, final_partitions, initial_partition, final_dist, func_history, num_iter, fpc = \
        fuzzy.cmeans(data=data, c=num_categories, m=2, error=0.005, maxiter=10000)
    return final_partitions, metadata, fpc


def display_categories(cat_data, base_image, filenames):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)]
    roi_memberships = np.argmax(cat_data, axis=0)
    overlay = im.copy()
    output = im.copy()
    roi_dims = (50, 50)
    print("Image Dims: ", im.shape)
    for i in range(len(filenames)):
        roi_loc = hg.parse_filename(filenames[i])
        tl_corner = (roi_loc['region'][0]*roi_dims[0] + roi_loc['offset'][0], roi_loc['region'][1]*roi_dims[1] + roi_loc['offset'][1])
        br_corner = (tl_corner[0] + roi_dims[0], tl_corner[1] + roi_dims[1])
        print("ITER: ", i, " LOC: ", roi_loc, " CORNER: ", br_corner)
        color = colors[roi_memberships[i]]
        cv2.rectangle(overlay, tl_corner, br_corner, color, -1)
        cv2.addWeighted(overlay, 0.01, output, 1 - 0.01, 0, output)
    cv2.imshow("ROI Membership", output)
    cv2.waitKey()
    raise NotImplementedError


if __name__ == '__main__':
    filename = r"C:\\Users\\HarrelsonT\\PycharmProjects\\HOGTest\\Spartan - Cell\\images_63780012_20180119130234_IMAG0002-100-2.JPG"
    im = cv2.imread(filename)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    final_partitions, metadata, fpc = fuzzy_c_means(5)
    display_categories(final_partitions, im, metadata)
