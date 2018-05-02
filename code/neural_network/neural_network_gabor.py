import sys
sys.path.append(r'../hog_generator')
import numpy as np
import gabor_threads_roi as hg
#from gabor_filter import GaborFilter
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
    dataPCA = data
    #dataPCA, pca, ss = hg.PCA(data, 9)


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


def predict(ann_loc, color_img, combined_filename):
    # load the weights and prepare objects
    ann_file = open(ann_loc, 'rb')
    ann = pickle.load(ann_file)
    ann_file.close()

    mask_filename = r"../mask_generator/gray-mask.png"
    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
    filters = hg.build_filters()

    all_data = hg.run_gabor(color_img, filters, mask, combined_filename, orientations=16, mode='validation')
    bands = all_data[:, 0]
    data = all_data[:, 1:-2]
    metadata = all_data[:, -2:]

    roi_predictions = np.zeros((color_img.shape[0], color_img.shape[1]))
    for i in range(data.shape[0]):
        y = int(metadata[i, 0:1][0])
        x = int(metadata[i, 1:2][0])

        roi_predictions[y:y+31, x:x+31] = ann.predict(data[i, :].reshape(1, -1))


    return roi_predictions


def view_predict(base_image, pixel_prediction):
    overlay = base_image.copy()
    colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
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
            data_loc = "../hog_generator/HOG_GABOR"
            pca, ss = train(data_loc)
        else:
            filename = "../image_subtractor/images/_usr_local_apps_scripts_bcj_webCam_images_64583391_20180124100038_IMAG1168-100-1168.JPG"
            #filename = "../image_subtractor/images/images_63816752_20180119161134_IMAG0190-100-190.JPG"

            img = cv2.imread(filename)
            assert img is not None
            gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pixel_predictions = predict("../neural_network/ann_1.pkl", img, filename)
            view_predict(img, pixel_predictions)
