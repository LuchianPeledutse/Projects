
import numpy as np

import torch
from torchvision import transforms
from torch.nn import functional as F
from torchvision.transforms import functional as TF

def geometric_augmentation(img):
    """
    description
    -----------
    applies geometric transformations for handwriting OCR.

    parameters
    ----------
    img:PIL image 
        pil image to apply transforms to

    returns
    -------
    list of transforms_images (in a PIL form) including the one present
    """
    #creating a list for storing transformed PIL images
    augmented_images = [img]
    #Move PIL image to tensor
    to_tensor_transform = transforms.ToTensor()
    img = to_tensor_transform(img)

    # applies a small rotation
    rotated = TF.rotate(img, angle=np.random.uniform(-7, 7))
    augmented_images.append(TF.to_pil_image(rotated))

    # applies a translation
    tx, ty = np.random.uniform(-0.03, 0.03), np.random.uniform(-0.03, 0.03)
    translated = TF.affine(img, angle=0, translate=(tx * img.shape[2], ty * img.shape[1]), scale=1.0, shear=0)
    augmented_images.append(TF.to_pil_image(translated))

    # applies a scaling
    scale = np.random.uniform(0.9, 1.1)
    scaled = TF.affine(img, angle=0, translate=(0, 0), scale=scale, shear=0)
    augmented_images.append(TF.to_pil_image(scaled))

    # applies a shear
    shear = np.random.uniform(-5, 5)
    sheared = TF.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=shear)
    augmented_images.append(TF.to_pil_image(sheared))

    # applies an elastic distortion (using torchvision built-in)
    elastic = transforms.ElasticTransform(alpha=35.0, sigma=5.0)(img)
    augmented_images.append(TF.to_pil_image(elastic))

    return augmented_images


def photometric_augmentation(img):
    """
    description
    -----------
    applies brightness, contrast, blur, and noise augmentations.

    parameters
    ----------
    img:PIL image 
        pil image to apply transforms to

    returns
    -------
    list of transforms_images (in a PIL form) including the one present
    """
    #creating a list for storing transformed PIL images
    augmented_images = [img]
    #Move PIL image to tensor
    to_tensor_transform = transforms.ToTensor()
    img = to_tensor_transform(img)

    # adjusting brightness
    bright = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.7, 1.3))
    augmented_images.append(TF.to_pil_image(bright))

    # adjusting contrast
    contrast = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.7, 1.3))
    augmented_images.append(TF.to_pil_image(contrast))

    # applying aussian blur
    blur = transforms.GaussianBlur(kernel_size=3, sigma=np.random.uniform(0.1, 3.5))(img)
    augmented_images.append(TF.to_pil_image(blur))

    # applying Gaussian noise
    noise = img + torch.randn_like(img) * 0.1
    noise = torch.clamp(noise, 0, 1)
    augmented_images.append(TF.to_pil_image(noise))

    # applying Color jitter (if color image)
    jitter = transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.2, hue=0.02)(img)
    augmented_images.append(TF.to_pil_image(jitter))

    return augmented_images




def structural_augmentation(img):
    """
    description
    -----------
    applies text-structure-related augmentations like erosion, cutout, and gradients.

    parameters
    ----------
    img:PIL image 
        pil image to apply transforms to

    returns
    -------
    list of transforms_images (in a PIL form) including the one present
    """
    #creating a list for storing transformed PIL images
    augmented_images = [img]
    #Move PIL image to tensor
    to_tensor_transform = transforms.ToTensor()
    img = to_tensor_transform(img)

    #applies erosion (simulates thinner strokes)
    kernel = torch.ones((1, 3, 3, 3), device=img.device) / 9
    eroded = F.conv2d(img.unsqueeze(0), kernel, padding=1).squeeze(0)
    augmented_images.append(TF.to_pil_image(eroded))

    #applies dilation (simulates thicker strokes)
    dilated = 1 - F.conv2d(1 - img.unsqueeze(0), kernel, padding=1).squeeze(0)
    augmented_images.append(TF.to_pil_image(dilated))

    # applies cutout / occlusion
    cutout = img.clone()
    h, w = img.shape[1:]
    for _ in range(np.random.randint(1, 3)):
        y, x = np.random.randint(0, h - 10), np.random.randint(0, w - 10)
        h_cut, w_cut = np.random.randint(50, 150), np.random.randint(50, 150)
        cutout[:, y:y + h_cut, x:x + w_cut] = 1.0  # white patch
    augmented_images.append(TF.to_pil_image(cutout))

    # adds background gradient (lighting)
    gradient = torch.linspace(0.8, 1.0, steps=h, device=img.device).view(1, h, 1)
    gradient_img = torch.clamp(img * gradient, 0, 1)
    augmented_images.append(TF.to_pil_image(gradient_img))

    return augmented_images



def synthetic_augmentation(img):
    """
    description
    -----------
    Blend with random paper textures or synthetic noise patterns.

    parameters
    ----------
    img:PIL image 
        pil image to apply transforms to

    returns
    -------
    list of transforms_images (in a PIL form) including the one present
    """
    #creating a list for storing transformed PIL images
    augmented_images = [img]
    #Move PIL image to tensor
    to_tensor_transform = transforms.ToTensor()
    img = to_tensor_transform(img)

    # Random background noise
    noise_bg = torch.rand_like(img) * 1.2 + 0.8  # light background
    overlay = torch.clamp(img * noise_bg, 0, 1)
    augmented_images.append(TF.to_pil_image(overlay))

    # Ink smudge simulation
    smudge = transforms.GaussianBlur(kernel_size=5, sigma=np.random.uniform(1.0, 5.5))(img)
    mixed = torch.clamp(img * 0.5 + smudge * 0.5, 0, 1)
    augmented_images.append(TF.to_pil_image(mixed))

    return augmented_images



def apply_transofrmation(img):
    """
    description
    -----------
    apply all of the above transformation to an image
    and return the list of transformations"""
    #create a list of all transformations
    all_transformations = [img]
    #add all of the transformations
    all_transformations += geometric_augmentation(img)[1:] #excluding the image itself (already present)
    all_transformations += photometric_augmentation(img)[1:]
    all_transformations += structural_augmentation(img)[1:]
    all_transformations += synthetic_augmentation(img)[1:]
    return all_transformations    