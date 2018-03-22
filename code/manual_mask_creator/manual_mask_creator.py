import cv2
import numpy as np

MOUSE_STATE = 'up'

# https://docs.opencv.org/3.1.0/d7/dfc/group__highgui.html#gga927593befdddc7e7013602bca9b079b0aa3536f83b6f48da5121041f58fc7a683
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html


# mouse callback function
def draw_circle(event,x,y,flags,param):
    global MOUSE_STATE
    if event == cv2.EVENT_LBUTTONDOWN:
        MOUSE_STATE = 'down'
    
    if event == cv2.EVENT_LBUTTONUP:
        MOUSE_STATE = 'up'
        
    if MOUSE_STATE == 'down':
        cv2.circle(img,(x,y),50,(255,0,0),-1)
        

# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
