from __future__ import division
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np
from builtins import map

try:
    # If you use vigra, we do special handling to preserve axistags
    import vigra
    _vigra_available = True
except ImportError:
    _vigra_available = False

# https://github.com/ilastik/lazyflow/blob/master/lazyflow/utility/blockwise_view.py
def blockwise_view(a, blockshape, aslist=False, require_aligned_blocks=True):
    """
    Return a 2N-D view of the given N-D array, rearranged so each ND block (tile)
    of the original array is indexed by its block address using the first N
    indexes of the output array.

    Note: This function is nearly identical to ``skimage.util.view_as_blocks()``, except:
          - "imperfect" block shapes are permitted (via require_aligned_blocks=False)
          - only contiguous arrays are accepted.  (This function will NOT silently copy your array.)
            As a result, the return value is *always* a view of the input.

    Args:
        a: The ND array
        blockshape: The tile shape

        aslist: If True, return all blocks as a list of ND blocks
                instead of a 2D array indexed by ND block coordinate.
        require_aligned_blocks: If True, check to make sure no data is "left over"
                                in each row/column/etc. of the output view.
                                That is, the blockshape must divide evenly into the full array shape.
                                If False, "leftover" items that cannot be made into complete blocks
                                will be discarded from the output view.

    Here's a 2D example (this function also works for ND):

    >>> a = numpy.arange(1,21).reshape(4,5)
    >>> print a
    [[ 1  2  3  4  5]
     [ 6  7  8  9 10]
     [11 12 13 14 15]
     [16 17 18 19 20]]
    >>> view = blockwise_view(a, (2,2), False)
    >>> print view
    [[[[ 1  2]
       [ 6  7]]
    <BLANKLINE>
      [[ 3  4]
       [ 8  9]]]
    <BLANKLINE>
    <BLANKLINE>
     [[[11 12]
       [16 17]]
    <BLANKLINE>
      [[13 14]
       [18 19]]]]
    Inspired by the 2D example shown here: http://stackoverflow.com/a/8070716/162094
    """
    assert a.flags['C_CONTIGUOUS'], "This function relies on the memory layout of the array."
    blockshape = tuple(blockshape)
    outershape = tuple(np.array(a.shape) // blockshape)
    view_shape = outershape + blockshape

    if require_aligned_blocks:
        assert (np.mod(a.shape, blockshape) == 0).all(), \
            "blockshape {} must divide evenly into array shape {}" \
                .format(blockshape, a.shape)

    # inner strides: strides within each block (same as original array)
    intra_block_strides = a.strides

    # outer strides: strides from one block to another
    inter_block_strides = tuple(a.strides * np.array(blockshape))

    # This is where the magic happens.
    # Generate a view with our new strides (outer+inner).
    view = np.lib.stride_tricks.as_strided(a, shape=view_shape,
                                           strides=(inter_block_strides + intra_block_strides))

    # Special handling for VigraArrays
    if _vigra_available and isinstance(a, vigra.VigraArray) and hasattr(a, 'axistags'):
        view_axistags = vigra.AxisTags([vigra.AxisInfo() for _ in blockshape] + list(a.axistags))
        view = vigra.taggedView(view, view_axistags)

    if aslist:
        return list(map(view.__getitem__, np.ndindex(outershape)))
    return view


def create_hog(region_image):
    raise NotImplementedError


def select_hog_regions(total_iamge, stability_mask, num_classes, region_size, offset_step_x=1, offset_step_y=1):
    print("Selecting ROIs")
    accepted_regions = -1

    # get regions
    for cur_offset_x in range(0, region_size[0], offset_step_x):
        for cur_offset_y in range(0, region_size[1], offset_step_y):
            tiled_view = blockwise_view(total_iamge[cur_offset_x:, cur_offset_y:], region_size, aslist=True, require_aligned_blocks=True)
            # throw away regions that are between multiple classes in the mask, or regions that are smaller than the region size
            index_to_exclude = []
            for tile_index in tiled_view.shape[0]:
                unique = np.unique(tiled_view[tile_index])
                if unique.shape[0] > 1 or unique.shape[1] > 1:
                    index_to_exclude.append(tile_index)
            tiled_view = np.delete(tiled_view, index_to_exclude)
            if accepted_regions == -1:
                accepted_regions = tiled_view
            else:
                accepted_regions = np.concatenate((accepted_regions, tiled_view))

    raise accepted_regions


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

