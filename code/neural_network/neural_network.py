import sys
sys.path.append(r'C:\\Users\\HarrelsonT\\PycharmProjects\\StreamGauge\\code\\hog_generator')
import numpy as np
import hog_generator as hg
import sklearn.neural_network as sk
import pickle
import time
from skimage.feature import hog
import cv2


def train(data_loc):
    # retrieve data from files
    data, metadata, bands = hg.load_hogs(data_loc)

    #build ANN object
    ann = sk.MLPClassifier(hidden_layer_sizes=100, activation='logistic', learning_rate_init=.01, max_iter=10000,
                           batch_size=200)   # lots of other options exist, check documentation
    # train the ANN
    ann.fit(data, bands.ravel())

    # save the ann
    timestr = '1' #time.strftime("%Y%m%d-%H%M%S")
    ann_file = open('../ann_' + timestr + '.pkl', 'wb')
    pickle.dump(ann, ann_file)
    ann_file.close()


def predict(ann_loc, img, roi_size, pixels_per_cell_list, orientations=9, cells_per_block=(2, 2), stride = 45):
    # load the weights and prepare objects
    ann_file = open(ann_loc, 'rb')
    ann = pickle.load(ann_file)
    ann_file.close()

    # produce all possible roi hogs and classify them
    roi_predictions = np.ones(((img.shape[0]-roi_size[1])//stride, (img.shape[1]-roi_size[0])//stride))
    roi_predictions = roi_predictions*-1    # set default value to -1 for clear indication of failures
    for y in range(0, roi_predictions.shape[0]):
        for x in range(0, roi_predictions.shape[1]):
            # pull roi from image
            cur_roi = img[y:y+roi_size[1], x:x+roi_size[0]]

            # create total hog feature of roi
            hog_info_total = []
            for pixels_per_cell in pixels_per_cell_list:
                hog_info = hg.make_hog_partial(cur_roi, orientations, pixels_per_cell, cells_per_block)
                hog_info_total.append(hog_info)
            hog_info_total = np.array(hog_info_total).flatten()

            # run hog through ann to get classification, and store it
            roi_predictions[y, x] = ann.predict(hog_info_total.reshape(1, -1))
            # if x % 100 == 0:
            print("ROI | x: ", x*stride, " y: ", y*stride, " predict: ", roi_predictions[y,x])

    # using roi classification, classify pixels
    pixel_predictions = np.zeros((img.shape[0], img.shape[1]))
    pixel_predictions = pixel_predictions*-1    # set default value to -1 for clear indication of failures
    # for each pixel
    for y in range(pixel_predictions.shape[0]):
        for x in range(pixel_predictions.shape[1]):
            # for each roi the pixel is within
            relevant_roi_loc = (max(0, y-roi_size[1]//2)//stride, min(roi_predictions.shape[0]*stride-1, y+roi_size[1]//2)//stride,
                                max(0, x-roi_size[0]//2)//stride, min(roi_predictions.shape[1]*stride-1, x+roi_size[0]//2)//stride)
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
                    prediction = round(total_predict/num_predict)
                else:
                    prediction = -1
            else:
                prediction = -1
            # prediction = round(np.mean(relevant_roi, dtype=np.float64))
            pixel_predictions[y,x] = prediction
            if x % 100 == 0:
                print("PIXEL | x: ", x, " y: ", y, " predict: ", prediction)
    return pixel_predictions


def view_predict(base_image, pixel_prediction):
    overlay = base_image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128),
              (153, 0, 76), (0, 153, 0)]

    for y in range(pixel_prediction.shape[0]):
        for x in range(pixel_prediction.shape[1]):
            if pixel_prediction[y,x] >= 0:
                color = colors[int(pixel_prediction[y,x])]
            else:
                color = (0, 0, 0)
            print("DISPLAY X: ", x, " Y: ", y)
            overlay[y,x] = color
    cv2.addWeighted(base_image, 0.8, overlay, 1 - 0.8, 0, overlay)
    cv2.imshow("Pixel Classification", overlay)
    cv2.waitKey()


if __name__ == '__main__':
    while True:
        user_input = input("Train or Predict: ")
        user_input = user_input.lower()
        if user_input == 'train':
            data_loc = r"C:\\Users\\HarrelsonT\\PycharmProjects\\StreamGauge\\code\\hog_generator\\HOG Files"
            train(data_loc)
        else:
            filename = r"C:\\Users\\HarrelsonT\\PycharmProjects\\HOGTest\\Spartan - Cell\\images_63780012_20180119130234_IMAG0002-100-2.JPG"
            img = cv2.imread(filename)
            gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pixel_predictions = predict("../ann_1.pkl", gs, (45, 45), [(3, 3), (5, 5), (9, 9)])
            view_predict(img, pixel_predictions)
