To Install:
Ensure that the folders image_subtractor, mask_generator, hog_generator, and neural_network are in the
same directory. Python3 must be installed on the device with OpenCV and Scikit, and NumPy modules also installed.

To Generate Mask:
1. Run image_subtractor.py
    a. Command line arguments may be given to specify filenames and other parameters. See documention
2. Run mask_generator.py

To Generate Training Data:
1. Ensure mask has been generated
2. Run gabor_threads_roi.py

To Train ANN:
1. Ensure training data exists
2. Run neural_network.py
3. When prompted, type "train" and hit enter

To Predict using ANN:
1. Ensure ANN is trained
2. Run neural_network.py
    a. You can train and predict the ANN without having to rerun the file, skip this step if this file is already
        running.
3. When prompted, type "predict" and hit enter
4. Predictions should appear as an image popup, and the function will return the data as an array (see documentation)


Tuning the parameters for the gabor filter can be done by modifying parameters for functions in the file
gabor_threads_roi.py. See documentation for function details.

