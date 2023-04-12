import caffe

import numpy as np

from PIL import Image

# Load image using PIL

img = Image.open('cat.jpg')

img_array = np.array(img)

# Define data augmentation transforms

transformer = caffe.io.Transformer({'data': (1,3,256,256)})

transformer.set_transpose('data', (2,0,1))

transformer.set_mean('data', np.array([104,117,123]))

transformer.set_raw_scale('data', 255)

transformer.set_channel_swap('data', (2,1,0))

# Create 20 augmented images

for i in range(20):

    # Apply random data augmentation

    transformed_image = transformer.preprocess('data', img_array)

    transformed_image = transformed_image[np.newaxis,:]

    

    # Save augmented image

    caffe.io.imsave(f'reproduced_photos/cat_augmented_{i+1}.jpg', transformer.deprocess('data', transformed_image[0]))

