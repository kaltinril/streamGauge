import sys
sys.path.append(r'../hog_generator')  # Allow importing code from the hog_generator folder
import numpy as np
import gabor_threads_roi as hg
import sklearn.neural_network as sk
import pickle
import cv2


def train(data_loc):
    """
    Load all Gabor filter feature data from the files in the data_loc folder,
    Train a Artifical Nerual Network with that data
    Save the trained model to a file ann_1.pk1

    :param data_loc:    Folder where the csv files are located
    """
    # retrieve data from files
    print("Loading hog data...")
    all_hogs = hg.load_hogs_csv(data_loc)

    # Remove band 3, as it is the "Unstable water region"
    rows_total = all_hogs.shape[0]
    all_hogs = all_hogs[(all_hogs[:, 0] != 3)]
    print("Removed band 3 rows", rows_total - all_hogs.shape[0])

    bands = all_hogs[:, 0]
    data = all_hogs[:, 1:-2]
    metadata = all_hogs[:, -2:]

    #build ANN object
    print("Training ANN...")
    ann = sk.MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation='tanh', learning_rate_init=0.01, max_iter=500000,
                           batch_size=2000, tol=.00000001, verbose=True, beta_1=0.9, beta_2=0.999, alpha=0.0001)   # lots of other options exist, check documentation
    # train the ANN
    ann.fit(data, bands)

    # save the ann
    timestr = '1' #time.strftime("%Y%m%d-%H%M%S")
    ann_file = open('ann_' + timestr + '.pkl', 'wb')
    pickle.dump(ann, ann_file)
    ann_file.close()


def predict(ann_loc, color_img, combined_filename, mask):
    """
    Load the trained ANN, run the Gabor histogram generation on the validation image, return the predicted band
    values at each ROI area

    :param ann_loc:             Location of the Trained NN file
    :param color_img:           The image to use to validate with
    :param combined_filename:   The filename of the color_img
    :param mask:                The mask that was used for the training, not really used but passed into Gabor
    :return:                    The predicted band values
    """
    # load the weights and prepare objects
    ann_file = open(ann_loc, 'rb')
    ann = pickle.load(ann_file)
    ann_file.close()

    # Generate the filters for the Gabor feature extraction
    filters = hg.build_filters()

    # Extract the feature histogram of gabor filters for each Region of Interest
    all_data = hg.run_gabor(color_img, filters, mask, combined_filename, orientations=16, mode='validation')
    bands = all_data[:, 0]
    data = all_data[:, 1:-2]
    metadata = all_data[:, -2:]

    # Place the predicted band value at the location in the image where that ROI came from
    roi_predictions = np.zeros((color_img.shape[0], color_img.shape[1]))
    for i in range(data.shape[0]):
        y = int(metadata[i, 0:1][0])
        x = int(metadata[i, 1:2][0])
        roi_predictions[y:y+31, x:x+31] = ann.predict(data[i, :].reshape(1, -1))

    return roi_predictions


def view_predict(base_image, pixel_prediction):
    """
    Overlay the predicted bands onto the original image for comparision and manual human evaluation

    :param base_image:          The image that was used to validate
    :param pixel_prediction:    The band predictions per ROI
    """
    overlay = base_image.copy()
    colors = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
    print("Start Display")
    for y in range(pixel_prediction.shape[0]):
        for x in range(pixel_prediction.shape[1]):
            if pixel_prediction[y,x] >= 0:
                color = colors[int(pixel_prediction[y, x])]
            else:
                color = (0, 0, 0)

            overlay[y, x] = color

    cv2.addWeighted(base_image, 0.7, overlay, 1 - 0.7, 0, overlay)  # apply region predictions with some transparency over the base image
    cv2.imshow("Pixel_Classification", overlay)
    cv2.waitKey()


if __name__ == '__main__':
    while True:
        user_input = input("Train or Predict: ")
        user_input = user_input.lower()
        if user_input == 'train':
            data_loc = "../hog_generator/HOG_GABOR"
            train(data_loc)
        else:
            filename = "../image_subtractor/images/images_63816752_20180119161134_IMAG0190-100-190.JPG"

            img = cv2.imread(filename)
            assert img is not None

            # Load the mask
            mask_filename = r"../mask_generator/gray-mask.png"
            mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)

            # Make sure the images are the correct dimensions, if not, resize to mask size
            if mask.shape[0:2] != img.shape[0:2]:
                print('Image size mismatch, resizing:', filename)
                img = hg.resize_image_to_mask(img, mask)

            pixel_predictions = predict("../neural_network/ann_1.pkl", img, filename, mask)
            view_predict(img, pixel_predictions)
