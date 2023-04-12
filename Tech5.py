import cv2

# load image using OpenCV

img = cv2.imread('cat.jpg')

# apply data augmentation to image using OpenCV

img_augmented = cv2.GaussianBlur(img, (5, 5), 0)

img_augmented = cv2.medianBlur(img_augmented, 5)

img_augmented = cv2.bilateralFilter(img_augmented, 9, 75, 75)

# save augmented image using OpenCV

cv2.imwrite('cat_augmented.jpg', img_augmented)

