import skimage.io as io

import skimage.color as color

import skimage.filters as filters

import skimage.exposure as exposure

import skimage.transform as transform

# define data augmentation techniques using scikit-image

def apply_augmentation(img_path, output_path, rotation_range=40, zoom_range=(0.8,1.2), flip_horiz=True, flip_vert=True):

    # load image

    img = io.imread(img_path)

    # apply random rotation

    img = transform.rotate(img, angle=np.random.uniform(-rotation_range, rotation_range), mode='reflect')

    # apply random zoom

    zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])

    img = transform.rescale(img, zoom_factor, mode='reflect')

    # apply random horizontal and vertical flipping

    if flip_horiz and np.random.random() < 0.5:

        img = np.fliplr(img)

    if flip_vert and np.random.random() < 0.5:

        img = np.flipud(img)

    # save augmented image

    io.imsave(output_path, img)

# apply data augmentation to input image using scikit-image

apply_augmentation('cat.jpg', 'cat_augmented.jpg')

# additional features

# load input image using scikit-image

img = io.imread('cat.jpg')

# apply gray scale to the image using scikit-image

gray_img = color.rgb2gray(img)

# apply histogram equalization to the image using scikit-image

eq_img = exposure.equalize_hist(gray_img)

# apply adaptive equalization to the image using scikit-image

adapteq_img = exposure.equalize_adapthist(gray_img, clip_limit=0.03)

# apply unsharp masking to the image using scikit-image

unsharp_masked_img = filters.unsharp_mask(img, radius=5, amount=2)

# save the processed images

io.imsave('cat_gray.jpg', gray_img)

io.imsave('cat_eq.jpg', eq_img)

io.imsave('cat_adapteq.jpg', adapteq_img)

io.imsave('cat_unsharp_masked.jpg', unsharp_masked_img)

