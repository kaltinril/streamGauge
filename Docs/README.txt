To Install:
Ensure that the folders image_subtractor, mask_generator, hog_generator, and neural_network are in the
same directory. Python3 must be installed on the device with OpenCV and Scikit, and NumPy modules also installed.

Setup and Data Prep:
* Images used for mask and training data should have the same camera location and dimensions.

* Make sure the images used to create the mask are photos taken sequentially in time from the same camera location (like
a timelapse photo sequence).

* Make sure all the images to be processed and turned into training data are from the same camera location. Try to
eliminate bad images (such as nighttime photos, when people are standing in front of the lense, etc) for best results.

* Retraining the ANN for each camera location will produce the best results.
------------------------------

To Generate Mask:
1. Run image_subtractor.py
    a. Command line arguments may be given to specify filenames and other parameters. See documention
2. Run mask_generator.py
    a. Command line arguments may be given to specify filenames and other parameters. See documention

To Generate Training Data:
1. Ensure mask has been generated
2. Run gabor_threads_roi.py

To Train ANN:
1. Ensure training data exists
2. Run neural_network_gabor.py
3. When prompted, type "train" and hit enter

To Predict using ANN:
1. Ensure ANN is trained
2. Run neural_network_gabor.py
    a. You can train and predict the ANN without having to rerun the file, skip this step if this file is already
        running.
3. When prompted, type "predict" and hit enter
4. Predictions should appear as an image popup, and the function will return the data as an array (see documentation)


Tuning the parameters for the gabor filter can be done by modifying parameters for functions in the file
gabor_threads_roi.py. See documentation for function details.

