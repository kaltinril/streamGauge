import cv2
import numpy as np
from tkinter.filedialog import askopenfilename
import sys
import os

MIN_RADIUS = 5
MAX_RADIUS = 100
CURRENT_RADIUS = 50

# https://docs.opencv.org/3.1.0/d7/dfc/group__highgui.html#gga927593befdddc7e7013602bca9b079b0aa3536f83b6f48da5121041f58fc7a683
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html


def restrict_radius(radius, max, min):

    if radius > max:
        radius = max

    if radius < min:
        radius = min

    return radius


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global MOUSE_STATE
    global CURRENT_RADIUS

    print(flags)

    color = (255, 0, 0)
    radius = CURRENT_RADIUS
    position = (x, y)
    thickness = -1  # positive = thickness of circle line, negative means fill

    if event == cv2.EVENT_MOUSEWHEEL:
        adjustment = 10
        if flags > 0:
            adjustment = -10

        CURRENT_RADIUS = restrict_radius(CURRENT_RADIUS + adjustment, MAX_RADIUS, MIN_RADIUS)

    if flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(OUTPUT_IMAGE, position, radius, color, thickness)


def select_image():
    filename = askopenfilename(initialdir="./",
                               title="Select file",
                               filetypes=(("jpeg files", "*.jpg"),
                                          ("png files", "*.png"),
                                          ("bmp files", "*.bmp"),
                                          ("all files", "*.*")))

    if filename is None:
        print("ERROR: No file selected")
        sys.exit(1)

    img = cv2.imread(filename)

    if img is None:
        print("ERROR: Invalid file format, unable to open!")
        sys.exit(1)

    return filename, img


# Create a black image, a window and bind the function to window
image_name, image = select_image()

OUTPUT_IMAGE = np.asarray(list(image))  # Deep clone/copy

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while(1):
    cv2.imshow('image', OUTPUT_IMAGE)
    if cv2.waitKey(20) & 0xFF == 27:
        break

path, filename = os.path.split(image_name)
filename, ext = os.path.splitext(filename)

output_name = filename + "_mask" + ext
output_name = os.path.join(path, output_name)

cv2.imwrite(output_name, OUTPUT_IMAGE)
cv2.destroyAllWindows()
