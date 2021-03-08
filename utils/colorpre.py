from PIL import Image
import numpy as np
import tifffile as tiff

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('L')
    color.putpalette(palette)
    return color


def blend_image(image, label, alpha = 0.6): #
    image = tiff.imread(image)
    image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
    image = image.convert('RGBA')
    label = label.convert('RGBA')
    out = Image.blend(image, label, alpha)
    return out