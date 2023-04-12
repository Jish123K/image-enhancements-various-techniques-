import numpy as np

import torch

from PIL import Image

from skimage import io

from skimage.transform import resize

from skimage.util import img_as_ubyte

# define data augmentation transforms using Deep Image Prior

def augment_image(img, num_iterations=1000, lr=0.01, sigma=0.1):

    img = img_as_ubyte(img)

    img = resize(img, (256, 256), anti_aliasing=True)

    img = img / 255.0

    img_tensor = torch.from_numpy(np.expand_dims(img.transpose((2, 0, 1)), axis=0)).float()

    img_tensor.requires_grad = True

    optimizer = torch.optim.Adam([img_tensor], lr=lr)

    for i in range(num_iterations):

        optimizer.zero_grad()

        loss = torch.norm(img_tensor - torch.randn_like(img_tensor) * sigma)

        loss.backward()

        optimizer.step()

    img = img_tensor.detach().numpy()[0].transpose((1, 2, 0))

    img = img * 255.0

    img = np.clip(img, 0, 255).astype(np.uint8)

    return Image.fromarray(img)

# load image using scikit-image

img = io.imread('cat.jpg')

# apply data augmentation to image using Deep Image Prior

img_augmented = augment_image(img)

# save augmented image using Pillow

img_augmented.save('cat_augmented.jpg')

