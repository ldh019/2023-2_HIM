from PIL import Image
from matplotlib import pyplot
import numpy as np


# Image Digitization
def get_pixel_data(name):
    im = Image.open(name, 'r')
    w, h = im.size
    result = list(im.getdata(0)) + list(im.getdata(1)) + list(im.getdata(2))
    result = np.array(result).reshape(3, h, w)
    return result


# Convolution Calculation
def calc_convolution(X, filters, stride=1, pad=1):
    _, h, w = X.shape
    filter_c, filter_h, filter_w = filters.shape

    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1

    # add padding to height and width.
    in_X = np.pad(X, [(0, 0), (pad, pad), (pad, pad)], 'constant')
    out = np.zeros((out_h, out_w))

    for c in range(filter_c):
        for h in range(out_h):  # slide the filter vertically.
            h_start = h * stride
            h_end = h_start + filter_h
            for w in range(out_w):  # slide the filter horizontally.
                w_start = w * stride
                w_end = w_start + filter_w
                # Element-wise multiplication.
                out[h, w] = np.sum(in_X[c, h_start:h_end, w_start:w_end] * filters[c])

    return out


# Draw Rectangle
def draw_square():
    print("hello")


name = 'image.png'
icon_name = 'target1.png'
digitized = get_pixel_data(name)
target = get_pixel_data(icon_name)
conv = calc_convolution(digitized, target)

# print(conv.flatten().sort()[:10])

pyplot.figure()
pyplot.imshow(conv)
pyplot.jet()
pyplot.colorbar()
pyplot.show()
