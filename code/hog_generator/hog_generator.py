import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np
import glob
import os
import errno
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import time


# need to clip total_image to be same size as stability mask before using this function
def create_hog_regions(color_image, stability_mask, image_filename, region_size, pixels_per_cell_list,
                       offset_step=(1, 1), region_threshold=0.9, orientations=9, cells_per_block=(2, 2),
                       banded=True):
    """
    This function takes an image, splits it into many regions of interest (ROIs), and then produces the histogram of
    oriented gradients (HOGs) for each ROI. If a ROI lies between categories when looking at the stability mask, it will
    be ignored and not saved. The method of culling data helps to only select data that clearly defines a type of object
    so the ANN can more accurately train as there is a greater difference in data between the categories of objects.

    :param color_image: A 2D array of pixels that will be converted to a GREYSCALE image (color is not accepted).
    :param stability_mask: A 2D array of values, each unique value is a category of object, so the value 0 may be land,
    while 1 may be water. This array must be the same shape as the total image.
    :param image_filename: A string representing the name of the image. This should match the name of the file the
    total_image was garnered from for the sake of being able to track the source of data for debugging, but it is not
    required. This parameter is used to generate the name of the files that contain the HOG data for the image.
    :param region_size: A 2 item tuple representing the width and height (in pixels) of the ROIs to be generated.
    :param offset_step: A 2 item tuple representing the number of pixels to shift the grid in the horizontal and
    vertical, used to split the image into ROI.
    :param region_threshold: A float between 0 and 1 representing the percent of a ROI that must be lies within a single
    category defined in the mask.
    :param orientations: An integer representing the number of bins the hog uses per cell, splitting up the orientations
    0-180 degrees into this number of bins.
    :param pixels_per_cell_list: A list of 2 item tuples representing the number of pixel wide and tall a cell is in
    the HOG algorithm
    :param cells_per_block: A 2 item tuple representing the cells wide and tall a bock is in the HOG algorithm
    :param banded: A boolean value representing if the stability mask has been banded, where each row in the 2D array
    has been averaged to as single category, so the algorithm used to split the image into ROI can take some shortcuts
    :return: Nothing, but saves the results of the HOG algorithm for each ROI to individual files in a folder
    """
    print("Creating ROI HOGs")

    image_grey = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    stability_mask = cv2.cvtColor(stability_mask, cv2.COLOR_BGR2GRAY)
    band_map = np.unique(stability_mask)
    # Asserts for debugging and enforcing data rules for parameters
    assert stability_mask.shape[0:2] == image_grey.shape, "Mask and Image different sizes, must be same size (excluding color channel)"
    assert region_size[0] <= image_grey.shape[1] and region_size[1] <= image_grey.shape[0], \
        "Region Size large than Image Dimensions"
    assert 0 < region_threshold <= 1.0, "Region Threshold outside of range [0,1)"
    for pixels_per_cell in pixels_per_cell_list:
        assert pixels_per_cell[0] > 0 and pixels_per_cell[1] > 0, "Pixels per Cell items must be positive"
        assert cells_per_block[0] > 0 and cells_per_block[1] > 0, "Cells per Block must be positive"
        assert pixels_per_cell[0]*cells_per_block[0] <= image_grey.shape[1] and pixels_per_cell[1] * cells_per_block[1] \
                                                                                 <= image_grey.shape[0], "Image too small for number of cells and blocks"
        assert offset_step[0] < region_size[0] and offset_step[1] < region_size[1], "Offset Step too large"

    # Create the folder to put the HOG files if none exists. Try except handles race condition, unlike straight makedirs
    try:
        os.makedirs("./HOG Files")
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Could not create or find HOG Files directory")
            raise

    total_calc = 0
    total_save = 0

    # shifts grid by offset to get slightly different pictures
    for cur_offset_x in range(0, region_size[0], offset_step[0]):
        for cur_offset_y in range(0, region_size[1], offset_step[1]):
            # looks at each tile of the grid with the current offset to produce ROIs
            for cur_region_y in range(0, image_grey.shape[0]//region_size[1]):
                for cur_region_x in range(0, image_grey.shape[1]//region_size[0]):
                    start_time = time.time()


                    hog_info_total = []
                    # Set initial band value to -1 to ensure it is obvoius if something goes wrong and band isnt set
                    # For some reason, np.savez does not always save numbers if they aernt in ndarrays
                    # so make band array for saving
                    band = np.ones((1,))
                    band *= -1

                    region_coords = (cur_offset_x + cur_region_x*region_size[0], cur_offset_y + cur_region_y*region_size[1])
                    masked_region = stability_mask[region_coords[1]:region_coords[1]+region_size[1],
                                                   region_coords[0]:region_coords[0]+region_size[0]]
                    image_region = image_grey[region_coords[1]:region_coords[1]+region_size[1],
                                              region_coords[0]:region_coords[0]+region_size[0]]

                    # this can be changed to return counts of each unique entry, so we can calculate percents
                    unique, unique_counts = np.unique(masked_region, return_counts=True)
                    sum_unique = region_size[0]*region_size[1]
                    unique_percents = unique_counts/sum_unique  # The percent of the region occupied by each type of mask
                    most_common_index = np.argmax(unique_counts)
                    band[0], = np.where(band_map == unique[most_common_index])
                    if unique_percents.max() < region_threshold or band[0] == 0:
                        # if banded then skip whole row, otherwise just skip current ROI
                        if banded:
                            break
                        else:
                            continue

                    # create hog for region
                    # unraveled shape=(n_blocks_y, n_blocks_x, cells_in_block_y, cells_in_block_x, orientations)
                    for pixels_per_cell in pixels_per_cell_list:
                        hog_info = make_hog_partial(image_region, orientations, pixels_per_cell, cells_per_block)

                        hog_info_total.append(hog_info)

                    # Create the histogram of colors for this region, we only need to do this once for the X/Y area
                    # color_image_region = color_image[region_coords[1]:region_coords[1] + region_size[1],
                    #                                 region_coords[0]:region_coords[0] + region_size[0]]
                    # color_hist = create_color_histogram(color_image_region)

                    # Add the 3 colors bins to the end of the hog_info_total array then convert and flatten
                    # color_hist = np.array(color_hist).flatten()
                    hog_info_total = np.array(hog_info_total).flatten()
                    # hog_info_total = np.append(hog_info_total, color_hist)

                    # format the string for the filename
                    new_filename = ("./HOG Files/%d_%d_%d_%d_%s_hogInfo.npz" % (cur_region_x, cur_region_y,
                                    cur_offset_x, cur_offset_y, image_filename))

                    total_calc += time.time() - start_time

                    assert band[0] >= 0

                    # save hog info, alongside other relevant info (pixel coords, base image file name)
                    start_time = time.time()
                    np.savez_compressed(new_filename, hog_info=hog_info_total, band=band)
                    total_save += time.time() - start_time

    print("Done Creating ROI HOGs")
    print("Total Calculation: ", total_calc)
    print("Total Save:", total_save)


def create_color_histogram(image, bins=8):
    """
    Calculates the Color histogram for an image and returns a 3 element Python array of arrays of size bins
    The size of each array element is [(blues),(greens),(reds)].  Where (color) length = bins count.

    :param image: any size image to perform the histogram on
    :param bins: (default: 8) - the number of separate slots/groups/bins.  8 means 0-7 is bin 1, 8-15 is bin 2, etc
    :return: A Python List of length 3, where each element contains an array of bin values for that color channel.
    """

    colors = []
    # Loop over the three colors (Blue, Green, Red) (OpenCV has this order)
    for i in range(0, 3):
        colors.append(cv2.calcHist([image], [i], None, [bins], [0, 256]))

    return colors


# From an image region, extracts the final single histogram for that region using HOG
def make_hog_partial(image_region, orientations, pixels_per_cell, cells_per_block):
    # unraveled shape=(n_blocks_y, n_blocks_x, cells_in_block_y, cells_in_block_x, orientations)
    hog_info = hog(image_region, orientations=orientations, pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block, visualise=False, block_norm='L2-Hys',
                   feature_vector=False)

    # list of blocks (each with a matrix of cells containing orientations)
    hog_info = np.reshape(hog_info, (hog_info.shape[0] * hog_info.shape[1], hog_info.shape[2],
                                     hog_info.shape[3], hog_info.shape[4]))

    # now flatten the cells to a list
    hog_info = np.reshape(hog_info, (hog_info.shape[0], hog_info.shape[1] * hog_info.shape[2],
                                     hog_info.shape[3]))

    # now average the cells
    hog_info = np.mean(hog_info, axis=1)

    # and average the blocks
    hog_info = np.mean(hog_info, axis=0)

    return hog_info


def load_hogs(folder_dir):
    """
    Retrieves the data from all .npz files (compressed or otherwise) in a folder, and returns the data and filenames

    :param folder_dir: A string that represents the path of the folder containing the .npz files
    :return: An ndarray containing the all the instances of the hog data, and a list containing the file names, in the
    same order
    """
    hog_list = []
    band_list = []
    file_list = []
    os.chdir(folder_dir)    # TODO: handle folder not found issues
    for file in glob.glob("*.npz"):
        hog_info, band = load_hog(file)
        hog_list.append(hog_info)
        band_list.append(band[0])
        file_list.append(file)
    return np.vstack(hog_list), file_list, np.vstack(band_list)


def load_hog(file):
    """
    Retrieves the data from a single .npz file

    :param file: A string representing the name of a .npz file, or the file object itself.
    :return: The ndarray contained in the file, decompressed, extracted, and ready to use.
    """
    loaded_file = np.load(file)
    return loaded_file['hog_info'], loaded_file['band']   # TODO: handle file not found issues


def parse_filename(filename):
    """
    Garners metadata for a HOG file that indicates what portion of the ROI is from, and which image file it was taken
    from

    :param filename: The string to be parsed, the name of a HOG file
    :return: a dictionary with the keys region and offset, the data of which are 2 item tuples representing the x and y
    dimensions of the region and offset.
    """
    splitstr = filename.split("_")
    assert len(splitstr) >= 4, "Filename is not in proper format"
    return {"region": (int(splitstr[0]), int(splitstr[1])), "offset": (int(splitstr[2]), int(splitstr[3]))}


def PCA(data_in, dim_out, standardize=True):
    """
    Uses sklearn's PCA function and StandardScaler function to transform a dataset to another subspace.

    :param data_in: An ndarray or matrix representing a collection of feature vectors to be transformed
    :param dim_out: An integer representing the number of dimensions in the subspace the feature vectors will be
    transformed into.
    :param standardize: A boolean value representing whether or not to standardize the data before running PCA.
    :return: Returns an ndarray having the same number of rows as data_in, but dim_out number of columns
    """
    data_out = data_in
    ss = StandardScaler()
    pca = decomposition.PCA(n_components=dim_out)
    if standardize:
        data_out = ss.fit_transform(data_in)
    data_out = pca.fit_transform(data_out)
    return data_out, pca, ss


if __name__ == '__main__':
    # Simple Example Use Scenario
    # filename_and_path = r"../image_subtractor/images/images_63796657_20180119143035_IMAG0089-100-89.JPG"
    #filename_and_path = r"../image_subtractor/images/images_64269229_20180122120038_IMAG0804-100-804.JPG"
    # filename_and_path = r"../image_subtractor/images/images_63816752_20180119161134_IMAG0190-100-190.JPG"
    filename_and_path = "../image_subtractor/images/_usr_local_apps_scripts_bcj_webCam_images_64583391_20180124100038_IMAG1168-100-1168.JPG"
    mask_filename = r"../mask_generator/gray-mask.png"
    path, filename = os.path.split(filename_and_path)
    filename_minus_ext, ext = os.path.splitext(filename)
    image_color = cv2.imread(filename_and_path)

    # real mask
    mask = cv2.imread(mask_filename)

    create_hog_regions(image_color, mask, filename_minus_ext, (45, 45), [(3, 3), (5, 5), (9, 9)], (9, 9), banded=False)

