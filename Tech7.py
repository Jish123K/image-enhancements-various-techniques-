import torch

import kornia

# Load image using PIL

pil_img = Image.open('cat.jpg')

# Convert PIL image to Torch tensor

img_tensor = kornia.image_to_tensor(pil_img).float() / 255.0

# Define data augmentation transforms

transforms = kornia.augmentation.Compose([

    kornia.augmentation.RandomAffine(degrees=40.0, translate=0.2, scale=(0.8, 1.2), shear=0.2),

    kornia.augmentation.RandomHorizontalFlip(p=0.5),

    kornia.augmentation.RandomVerticalFlip(p=0.5),

    kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

    kornia.augmentation.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),

])

# Augment the image 20 times

for i in range(20):

    # Apply random data augmentation

    augmented_tensor = transforms(img_tensor.unsqueeze(0))

    

    # Convert tensor to PIL image and save

    augmented_pil = kornia.tensor_to_image(augmented_tensor.squeeze(0))

    augmented_pil.save(f'reproduced_photos/cat_augmented_{i+1}.jpg')

