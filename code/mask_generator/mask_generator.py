import cv2

averaged_image = cv2.imread("../image_subtractor/all_combined.png")

print(averaged_image.shape)

cv2.imshow('image', averaged_image)
cv2.waitKey()


# Cleanup
cv2.destroyAllWindows()
