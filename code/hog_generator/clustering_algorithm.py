import numpy as np
import skfuzzy.cluster as fuzzy
import hog_generator as hg
import cv2
# import matplotlib.pyplot as plt
import sklearn.cluster as sk
import pickle

def fuzzy_c_means(num_categories):
    data, metadata, labels = hg.load_hogs('C:\\Users\\HarrelsonT\\PycharmProjects\\StreamGauge\\code\\hog_generator\\HOG Files')
    #data = hg.PCA(data, 15)
    # data must be of size (S, N) where S is the number of features in a vector, and N is the number of feature vectors
    data = data.T
    centroids, final_partitions, initial_partition, final_dist, func_history, num_iter, fpc = \
        fuzzy.cmeans(data=data, c=num_categories, m=-2, error=0.00005, maxiter=100000, init=None)
    return final_partitions, metadata, fpc


def k_means_predict(kmeans_loc, color_img, roi_size, pixels_per_cell_list, orientations=9, cells_per_block=(2, 2), stride=45):
    kmeans_file = open(kmeans_loc, 'rb')
    kmeans = pickle.load(kmeans_file)
    kmeans_file.close()

    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # produce all possible roi hogs and classify them
    roi_predictions = np.ones(((img.shape[0] - roi_size[1]) // stride, (img.shape[1] - roi_size[0]) // stride))
    roi_predictions = roi_predictions * -1  # set default value to -1 for clear indication of failures
    for y in range(0, roi_predictions.shape[0]):
        for x in range(0, roi_predictions.shape[1]):
            # pull roi from image
            cur_roi = img[y:y + roi_size[1], x:x + roi_size[0]]
            cur_roi_color = color_img[y:y + roi_size[1], x:x + roi_size[0]]

            # create total hog feature of roi
            hog_info_total = []
            for pixels_per_cell in pixels_per_cell_list:
                hog_info = hg.make_hog_partial(cur_roi, orientations, pixels_per_cell, cells_per_block)
                hog_info_total.append(hog_info)

            # color_hist = hg.create_color_histogram(cur_roi_color)

            # Add the 3 colors bins to the end of the hog_info_total array then convert and flatten
            # color_hist = np.array(color_hist).flatten()
            hog_info_total = np.array(hog_info_total).flatten()
            # hog_info_total = np.append(hog_info_total, color_hist)

            # run hog through ann to get classification, and store it
            roi_predictions[y, x] = kmeans.predict(hog_info_total.reshape(1, -1))
            # if x % 100 == 0:
            print("ROI | x: ", x * stride, " y: ", y * stride, " predict: ", roi_predictions[y, x])

    # using roi classification, classify pixels
    pixel_predictions = np.zeros((img.shape[0], img.shape[1]))
    pixel_predictions = pixel_predictions * -1  # set default value to -1 for clear indication of failures
    # for each pixel
    for y in range(pixel_predictions.shape[0]):
        for x in range(pixel_predictions.shape[1]):
            # for each roi the pixel is within
            relevant_roi_loc = (max(0, y - roi_size[1] // 2) // stride,
                                min(roi_predictions.shape[0] * stride - 1, y + roi_size[1] // 2) // stride,
                                max(0, x - roi_size[0] // 2) // stride,
                                min(roi_predictions.shape[1] * stride - 1, x + roi_size[0] // 2) // stride)
            if relevant_roi_loc[0] < roi_predictions.shape[0] and relevant_roi_loc[2] < roi_predictions.shape[1]:
                if relevant_roi_loc[0] != relevant_roi_loc[1] and relevant_roi_loc[2] != relevant_roi_loc[3]:
                    relevant_roi = roi_predictions[relevant_roi_loc[0]:relevant_roi_loc[1],
                                   relevant_roi_loc[2]:relevant_roi_loc[3]]
                else:
                    relevant_roi = np.array([roi_predictions[relevant_roi_loc[0], relevant_roi_loc[2]]])
                relevant_roi = relevant_roi.flatten()
                total_predict = 0
                num_predict = 0
                for p in relevant_roi:
                    if p >= 0:
                        total_predict += p
                        num_predict += 1
                if num_predict > 0:
                    prediction = round(total_predict / num_predict)
                else:
                    prediction = -1
            else:
                prediction = -1
            # prediction = round(np.mean(relevant_roi, dtype=np.float64))
            pixel_predictions[y, x] = prediction
            if x % 100 == 0:
                print("PIXEL | x: ", x, " y: ", y, " predict: ", prediction)
    return pixel_predictions


def k_means(num_categories, data_loc):
    data, metadata, labels = hg.load_hogs(data_loc)
    kmeans = sk.KMeans(n_clusters=num_categories, init="k-means++", max_iter=10000, n_jobs=-1)

    kmeans_info = kmeans.fit(data)

    timestr = '1'  # time.strftime("%Y%m%d-%H%M%S")
    kmeans_file = open('../kmeans_' + timestr + '.pkl', 'wb')
    pickle.dump(kmeans, kmeans_file)
    kmeans_file.close()
    return kmeans_info.labels_, metadata


def view_predict(base_image, pixel_prediction):
    overlay = base_image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128),
              (153, 0, 76), (0, 153, 0)]

    for y in range(pixel_prediction.shape[0]):
        for x in range(pixel_prediction.shape[1]):
            if pixel_prediction[y,x] >= 0:
                color = colors[int(pixel_prediction[y, x])]
            else:
                color = (0, 0, 0)
            print("DISPLAY X: ", x, " Y: ", y)
            overlay[y, x] = color
    cv2.addWeighted(base_image, 0.8, overlay, 1 - 0.8, 0, overlay)  # apply region predictions with some transparency over the base image
    cv2.imshow("Pixel Classification", overlay)
    cv2.waitKey()


def display_categories(cat_data, base_image, filenames):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128), (153, 0, 76), (0, 153, 0)]
    if len(cat_data.shape) == 1:    # if not fuzzy, membership will just declare the categories, not give confidence of each cat
        roi_memberships = cat_data
    else:
        roi_memberships = np.argmax(cat_data, axis=0)
    overlay = base_image.copy()
    output = base_image.copy()
    roi_dims = (50, 50)
    print("Image Dims: ", base_image.shape)
    for i in range(len(filenames)):
        overlay = output.copy()
        roi_loc = hg.parse_filename(filenames[i])
        # for whatever reason, cv2 holds x and y opposite of np, or at least opposite of the way it was stored/retrieved
        tl_corner = (roi_loc['region'][0]*roi_dims[0] + roi_loc['offset'][0], roi_loc['region'][1]*roi_dims[1] + roi_loc['offset'][1])
        br_corner = (tl_corner[0] + roi_dims[0], tl_corner[1] + roi_dims[1])
        print("ITER: ", i, " LOC: ", roi_loc, " CORNER: ", br_corner, " CAT: ", roi_memberships[i])
        color = colors[roi_memberships[i]]
        cv2.rectangle(overlay, tl_corner, br_corner, color, -1)
        cv2.addWeighted(overlay, 0.8, output, 1 - 0.8, 0, output)
    cv2.addWeighted(base_image, 0.6, output, 1 - 0.6, 0, output)
    cv2.imshow("ROI Membership", output)
    #cv2.imshow("ROI Membership", base_image)
    cv2.waitKey()


if __name__ == '__main__':
    #filename = r"../image_subtractor/images/images_63796657_20180119143035_IMAG0089-100-89.JPG"
    filename = r"../image_subtractor/images/images_64273512_20180122124538_IMAG0810-100-810.JPG"
    img = cv2.imread(filename)
    assert img is not None

    while True:
        user_input = input("Train or Predict: ")
        user_input = user_input.lower()
        if user_input == 'train':
            data_loc = r"C:\\Users\\HarrelsonT\\PycharmProjects\\StreamGauge\\code\\hog_generator\\HOG Files"
            kmeans_partition, metadata = k_means(3, data_loc)
            #display_categories(kmeans_partition, img, metadata)
        else:
            gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pixel_predictions = k_means_predict("../ann_1.pkl", img, (45, 45), [(3, 3), (5, 5), (9, 9)])
            view_predict(img, pixel_predictions)
    """
    filename = r"C:/Users/HarrelsonT/PycharmProjects/StreamGauge/code/image_subtractor/images/images_63796657_20180119143035_IMAG0089-100-89.JPG"
    im = cv2.imread(filename)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    final_partitions, metadata, fpc = fuzzy_c_means(5)
    # kmeans_partition = k_means(3)
    display_categories(final_partitions, im, metadata)
    # display_categories(kmeans_partition, im, metadata)
    """
