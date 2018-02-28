import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np
import pickle


# need to clip total_image to be same size as stability mask before using this function
def create_hog_regions(total_iamge, stability_mask, image_filename, region_size, offset_step_x=1, offset_step_y=1,
                       region_threshold=0.8, orientations=8, pixels_per_cell=(4,4), cells_per_block=(8,8)):
    print("Creating ROI HOGs")

    roi_file = open(image_filename + "_hog.data", "wb")
    # shifts grid by offset to get slightly different pictures
    for cur_offset_x in range(0, region_size[0], offset_step_x):
        for cur_offset_y in range(0, region_size[1], offset_step_y):
            # looks at each tile of the grid with the current offset
            for cur_region_y in range(0, total_iamge.shape[1]//region_size[1]):
                for cur_region_x in range(0, total_iamge.shape[0]//region_size[0]):
                    region_coords = (cur_offset_x + cur_region_x*region_size[0], cur_offset_y + cur_region_y*region_size[1])
                    """
                    if region_coords[2] < total_iamge.shape[0] and region_coords[3] < total_iamge.shape[1]:
                        continue
                    """
                    masked_region = stability_mask[region_coords[0]:region_coords[0]+region_size[0],
                                                   region_coords[1]:region_coords[1]+region_size[1]]
                    image_region = total_iamge[region_coords[0]:region_coords[0]+region_size[0],
                                               region_coords[1]:region_coords[1]+region_size[1]]

                    # this can be changed to return counts of each unique entry, so we can calculate percents
                    unique, unique_counts = np.unique(masked_region, return_counts=True)
                    sum_unique = region_size[0]*region_size[1]
                    unique_percents = unique_counts/sum_unique  # The percent of the region occupied by each type of mask
                    if unique_percents.max() < region_threshold:
                        break   # we can break this loop, as due to banding, all regions in this row have same issue
                    """
                    if unique.shape[0] > 1 or unique.shape[1] > 1:  # if there is >1 type of mask category in region
                        region_coords += region_size
                        continue
                    """

                    # create hog for region
                    hog_info = hog(image_region, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                   cells_per_block=cells_per_block, visualise=False, block_norm='L2-Hys')

                    # save hog info, alongside other relevant info (pixel coords, base image file name)
                    roi_info = (hog_info.dumps(), region_coords)
                    pickle.dump(roi_info, roi_file)

    roi_file.close()
    print("Done Creating ROI HOGs")


if __name__ == '__main__':
    filename = r"C:\\Users\\HarrelsonT\\PycharmProjects\\HOGTest\\Spartan - Cell\\images_63780012_20180119130234_IMAG0002-100-2.JPG"
    im = cv2.imread(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # make a fake mask for now.
    mask = np.ones((im.shape[0]-20, im.shape[1]-20))     # 10 pixel border on either end = 20 pixels removed from both dims
    # zero out upper right triangle to create edge to check if algorithm throws away regions correctly
    mask[0:mask.shape[0]//2, 0:-1] = 0
    # remove same pixel border from image
    im = im[10: im.shape[0]-20, 10:im.shape[1]-20]
    create_hog_regions(im, mask, 'images_63780012_20180119130234_IMAG0002-100-2', (200, 200), 50, 50)

    """
    filename = r"C:\\Users\\HarrelsonT\\PycharmProjects\\HOGTest\\Spartan - Cell\\images_63780012_20180119130234_IMAG0002-100-2.JPG"
    im = cv2.imread(filename)

    # convert color to greyscale, hog function doesnt accept color, despite saying otherwise in documentation
    gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    print(im.shape)
    image = gr

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(8, 8), visualise=True, block_norm='L2-Hys')

    fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)

    ax[0].axis('off')
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 20))

    ax[1].axis('off')
    ax[1].imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax[1].set_title('Histogram of Oriented Gradients')
    ax[1].set_adjustable('box-forced')
    fig.show()
    """

