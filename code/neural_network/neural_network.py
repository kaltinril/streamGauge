import sys
sys.path.append(r'../hog_generator')
import numpy as np
import hog_generator as hg
from gabor_filter import GaborFilter
import sklearn.neural_network as sk
import pickle
import time
from skimage.feature import hog
import cv2


def train(data_loc):
    # retrieve data from files
    print("Loading hog data...")
    all_hogs = hg.load_hogs_csv(data_loc)
    rows_total = all_hogs.shape[0]
    all_hogs = all_hogs[(all_hogs[:, 0] != 3)]

    print("Removed band 3 rows", rows_total - all_hogs.shape[0])

    bands = all_hogs[:, 0]
    data = all_hogs[:, 1:-2]
    metadata = all_hogs[:, -2:]

    print("Performing PCA...")
    dataPCA, pca, ss = hg.PCA(data, 9)

    #build ANN object
    print("Training ANN...")
    ann = sk.MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation='tanh', learning_rate_init=0.01, max_iter=500000,
                           batch_size=2000, tol=.00000001, verbose=True, beta_1=0.9, beta_2=0.999, alpha=0.0001)   # lots of other options exist, check documentation
    # train the ANN
    ann.fit(dataPCA, bands)

    # predictions = ann.predict(dataPCA)
    # for p in predictions:
    #     print(p)

    # save the ann
    timestr = '1' #time.strftime("%Y%m%d-%H%M%S")
    ann_file = open('ann_' + timestr + '.pkl', 'wb')
    pickle.dump(ann, ann_file)
    ann_file.close()
    return pca, ss


def predict(ann_loc, color_img, roi_size, pixels_per_cell_list, orientations=9, cells_per_block=(2, 2), stride=5, pca=None, ss=None):
    # load the weights and prepare objects
    ann_file = open(ann_loc, 'rb')
    ann = pickle.load(ann_file)
    ann_file.close()

    # Run a Gabor filter on the image to make the features turn out easier for hog detection
    gabor = GaborFilter()  # Use the default values
    color_img = gabor.process_threaded(color_img)

    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # produce all possible roi hogs and classify them
    roi_predictions = np.ones(((img.shape[0]-roi_size[1])//stride, (img.shape[1]-roi_size[0])//stride))
    roi_predictions = roi_predictions*-1    # set default value to -1 for clear indication of failures
    for y in range(0, roi_predictions.shape[0]):
        for x in range(0, roi_predictions.shape[1]):
            # pull roi from image
            cur_roi = img[y:y+roi_size[1], x:x+roi_size[0]]
            cur_roi_color = color_img[y:y + roi_size[1], x:x + roi_size[0]]

            # create total hog feature of roi
            hog_info_total = []
            for pixels_per_cell in pixels_per_cell_list:
                hog_info = hg.make_hog_partial(cur_roi, orientations, pixels_per_cell, cells_per_block)
                hog_info_total.append(hog_info)

            color_hist = hg.create_color_histogram(cur_roi_color)

            # Add the 3 colors bins to the end of the hog_info_total array then convert and flatten
            color_hist = np.array(color_hist).flatten()
            hog_info_total = np.array(hog_info_total).flatten().reshape(1, -1)
            hog_info_total = np.append(hog_info_total, color_hist).reshape(1, -1)
            hog_info_total = ss.transform(hog_info_total)
            hog_info_total = pca.transform(hog_info_total)

            # run hog through ann to get classification, and store it
            roi_predictions[y, x] = ann.predict(hog_info_total.reshape(1, -1))
            if roi_predictions[y, x] != 1:
                print("ROI | x: ", x*stride, " y: ", y*stride, " predict: ", roi_predictions[y, x])

    # using roi classification, classify pixels
    pixel_predictions = np.ones((img.shape[0], img.shape[1]))
    pixel_predictions = pixel_predictions*-1    # set default value to -1 for clear indication of failures
    # for each pixel
    for y in range(pixel_predictions.shape[0]):
        for x in range(pixel_predictions.shape[1]):
            # for each roi the pixel is within
            relevant_roi_loc = (int(max(0, y-roi_size[1]/2)//stride), int(min(roi_predictions.shape[0]*stride-1, y+roi_size[1]/2)//stride),
                                int(max(0, x-roi_size[0]/2)//stride), int(min(roi_predictions.shape[1]*stride-1, x+roi_size[0]/2)//stride))
            if relevant_roi_loc[0] < roi_predictions.shape[0] and relevant_roi_loc[2] < roi_predictions.shape[1]:
                if relevant_roi_loc[0] != relevant_roi_loc[1] and relevant_roi_loc[2] != relevant_roi_loc[3]:
                    relevant_roi = roi_predictions[relevant_roi_loc[0]:relevant_roi_loc[1],
                                                   relevant_roi_loc[2]:relevant_roi_loc[3]]
                else:
                    relevant_roi = np.array([roi_predictions[relevant_roi_loc[0], relevant_roi_loc[2]]])
                relevant_roi = relevant_roi.flatten()
                u, uc = np.unique(relevant_roi, return_counts=True)
                prediction = u[np.argmax(uc)]
                # total_predict = 0
                # num_predict = 0
                # for p in relevant_roi:
                #     if p >= 0:
                #         total_predict += p
                #         num_predict += 1
                # if num_predict > 0:
                #     prediction = round(total_predict/num_predict)
                # else:
                #     prediction = -1
            else:
                prediction = -1
            # prediction = round(np.mean(relevant_roi, dtype=np.float64))
            pixel_predictions[y, x] = prediction
            if prediction != 1:
                print("PIXEL | x: ", x, " y: ", y, " predict: ", prediction)
    return pixel_predictions


def view_predict(base_image, pixel_prediction):
    overlay = base_image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128),
              (153, 0, 76), (0, 153, 0)]
    print("Start Display")
    for y in range(pixel_prediction.shape[0]):
        for x in range(pixel_prediction.shape[1]):
            if pixel_prediction[y,x] >= 0:
                color = colors[int(pixel_prediction[y, x])]
            else:
                color = (0, 0, 0)
            #print("DISPLAY X: ", x, " Y: ", y)
            overlay[y, x] = color
    cv2.addWeighted(base_image, 0.8, overlay, 1 - 0.8, 0, overlay)  # apply region predictions with some transparency over the base image
    cv2.imshow("Pixel Classification", overlay)
    cv2.waitKey()


if __name__ == '__main__':
    pca = None
    ss = None
    while True:
        user_input = input("Train or Predict: ")
        user_input = user_input.lower()
        if user_input == 'train':
            data_loc = "../hog_generator/HOG_Files"
            pca, ss = train(data_loc)
        else:
            filename = "../image_subtractor/images/_usr_local_apps_scripts_bcj_webCam_images_64583391_20180124100038_IMAG1168-100-1168.JPG"
            #filename = "../image_subtractor/images/images_63816752_20180119161134_IMAG0190-100-190.JPG"

            img = cv2.imread(filename)
            assert img is not None
            gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pixel_predictions = predict("../neural_network/ann_1.pkl", img, (12, 12), [(3, 3)], stride=3, pca=pca, ss=ss)
            view_predict(img, pixel_predictions)
