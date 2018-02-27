import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np
import pickle


def create_hog(region_image):
    raise NotImplementedError

# need to clip total_image to be same size as stability mask before using this function
def select_hog_regions(total_iamge, stability_mask, image_filename, region_size, offset_step_x=1, offset_step_y=1):
    print("Selecting ROIs")

    roi_file = open("hog.data", "w+")
    for cur_offset_x in range(0, region_size[0], offset_step_x):
        for cur_offset_y in range(0, region_size[1], offset_step_y):
            region_coords = (cur_offset_x, cur_offset_y, cur_offset_x+region_size[0], cur_offset_y+region_size[1])
            # for each region in the image, starting at the offset X and Y index
            while region_coords[2] < total_iamge.shape[0] and region_coords[3] < total_iamge.shape[1]:
                masked_region = stability_mask[region_coords[0]:region_coords[2], region_coords[1]:region_coords[3]]
                image_region = total_iamge[region_coords[0]:region_coords[2], region_coords[1]:region_coords[3]]

                # this can be changed to return counts of each unique entry, so we can calculate percents
                unique = np.unique(masked_region)
                if unique.shape[0] > 1 or unique.shape[1] > 1:  # if there is >1 type of mask category in region
                    region_coords += region_size
                    continue

                # create hog for region
                hog_info = create_hog(image_region)

                # save hog info, alongside other relevant info (pixel coords, base image file name)
                roi_info = (hog_info.dumps(), region_coords, filename)
                pickle.dump(roi_info, roi_file)

                # increment to next region, no overlap of regions within this loop
                region_coords += region_size

    roi_file.close()
    raise NotImplementedError


if __name__ == '__main__':
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

