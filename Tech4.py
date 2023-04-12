import mahotas as mh

import numpy as np

# load image using Mahotas

img = mh.imread('cat.jpg')

# apply data augmentation to image using Mahotas

img_augmented = mh.gaussian_filter(img, 1.5)

img_augmented = mh.median_filter(img_augmented, 3)

img_augmented = mh.stretch(img_augmented, 0, 255)

# save augmented image using Mahotas

mh.imsave('cat_augmented.jpg', img_augmented)

# additional features

# calculate the mean and standard deviation of the image

img_mean = np.mean(img)

img_std = np.std(img)

# apply a contrast stretch to the image based on its mean and standard deviation

img_stretch = mh.stretch(img, img_mean - img_std, img_mean + img_std)

# apply a gamma correction to the image

img_gamma = mh.gamma_correct(img, 1.5)

# save the contrast stretched and gamma corrected images

mh.imsave('cat_stretched.jpg', img_stretch)

mh.imsave('cat_gamma.jpg', img_gamma)

