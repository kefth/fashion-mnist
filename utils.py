import torch
import torch.nn as nn
from PIL import Image, ImageOps
import numpy as np

def get_conv_out(layer, input_size):
    w, h = input_size
    F = layer.kernel_size
    S = layer.stride
    P = layer.padding
    w2= (w-F[0]+2*P[0])/S[0]+1
    h2 =(h-F[1]+2*P[1])/S[1]+1
    return w2,h2

def get_pool_out(layer,input_size):
    w, h = input_size
    F = layer.kernel_size
    S = layer.stride
    P = layer.padding
    w2 = (w-F)/S+1
    h2 = (h-F)/S+1
    return w2,h2

def calculate_feature_size(model, input_size):
    for layer in model:
        if type(layer) == nn.Conv2d:
            input_size = get_conv_out(layer, input_size)
        elif type(layer) == nn.MaxPool2d:
            input_size = get_pool_out(layer, input_size)
    return input_size

def get_multilabel_accuracy(pred, target):
    """ Calculate multilabel accuracy.

        Turn prediction tensor in binary. Compare with target.
        Calculate common elements. To be used for calculating running
        accuracy and total accuracy in training.
    """
    pred = pred > 0.5
    r = (pred == target.byte())
    acc = r.float().cpu().sum().data[0]
    return acc/(pred.size()[1]*pred.size()[0])

def save_model(model_state, filename):
    """ Save model """
    # TODO: add it as checkpoint
    torch.save(model_state,filename)

class RandomVerticalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if np.random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

class RandomRotation(object):
    """Rotate PIL.Image randomly (90/180/270 degrees)with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be rotated.
        Returns:
            PIL.Image: Randomly rotated image.
        """
        if np.random.random() < 0.5:
            deg = np.random.randint(1,3)*90.
            return img.rotate(deg)
        return img

class RandomTranslation(object):
    """Translates PIL.Image randomly (0-10 pixels) with a probability of 0.5."""

    def __init__(self,max_vshift=10, max_hshift=10):
        self.max_vshift = max_vshift
        self.max_hshift = max_hshift

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be translated.
        Returns:
            PIL.Image: Randomly translated image.
        """
        if np.random.random() < 0.5:
            hshift = np.random.randint(-self.max_hshift,self.max_hshift)
            vshift = np.random.randint(-self.max_vshift,self.max_vshift)
            return img.transform(img.size, Image.AFFINE, (1, 0, hshift, 0, 1, vshift))
        return img
