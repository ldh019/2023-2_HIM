from PIL import Image
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np


def my_padding(img, shape):
    pad_sizeY, pad_sizeX = int(shape[0] / 2), int(shape[1] / 2)
    res = np.pad(img, [(pad_sizeY, shape[0] - pad_sizeY), (pad_sizeX, shape[1] - pad_sizeX)], 'constant')
    return res


def my_kernel(shape, sigma):
    a = shape[0]
    b = shape[1]
    y, x = np.ogrid[-b:b + 1, -a:a + 1]
    gaus_kernel = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    sum = gaus_kernel.sum()
    gaus_kernel /= sum
    return gaus_kernel


def my_filtering(img, kernel):
    row, col = len(img), len(img[0])
    ksizeY, ksizeX = kernel.shape[0], kernel.shape[1]

    pad_image = my_padding(img, (ksizeY, ksizeX))
    filtered_img = np.zeros((row, col), dtype=np.float32)

    for i in range(row):
        for j in range(col):
            filtered_img[i, j] = np.sum(
                np.multiply(kernel, pad_image[i:i + ksizeY, j:j + ksizeX]))

    return filtered_img


def get_pixel_data(name):
    gray = np.array(Image.open(name).convert("L"))
    h, w = gray.shape
    gaus1 = my_kernel((5, 5), 1.6)
    gaus2 = my_kernel((5, 5), 1)
    gaussian1 = my_filtering(gray, gaus1)
    gaussian2 = my_filtering(gray, gaus2)
    DoG = np.zeros_like(gray)
    for i in range(h):
        for j in range(w):
            DoG[i][j] = float(gaussian1[i][j]) - float(gaussian2[i][j])
    result = np.array(DoG).reshape(1, h, w)

    result = (result - np.min(result)) / (np.max(result) - np.min(result))

    return result


def get_pixel_data_target(name):
    result = get_pixel_data(name)
    result = np.flip(result)
    return result


def calc_convolution(X, filters):
    _, h, w = X.shape
    filters = np.flip(filters)
    filter_c, filter_h, filter_w = filters.shape

    out_h = h
    out_w = w

    in_X = my_padding(X[0], (filter_h, filter_w))
    out = np.zeros((out_h, out_w))

    for c in range(filter_c):
        for h in range(out_h):
            h_start = h
            h_end = h_start + filter_h
            for w in range(out_w):
                w_start = w
                w_end = w_start + filter_w
                out[h, w] = np.sum(in_X[h_start:h_end, w_start:w_end] * filters[c])

    return out


def firstRgbResult(name, icon_name):
    image = np.array(Image.open(name).convert("L"))
    target = np.array(Image.open(icon_name).convert("L"))
    target = np.flip(target)

    conv = calc_convolution(image, target)

    plt.figure()
    plt.imshow(conv, cmap='jet')
    plt.colorbar()
    plt.show()


def finalResult(name, icon_name):
    origin = np.array(Image.open(name).convert("L"))
    digitized = get_pixel_data(name)
    target = get_pixel_data_target(icon_name)
    conv = calc_convolution(digitized, target)

    plt.figure()
    plt.imshow(digitized[0], cmap='gray')
    plt.colorbar()
    plt.show()

    fig, ax = plt.subplots()
    ax.imshow(origin, cmap='gray')
    indices = np.argwhere(conv >= np.max(conv) * .9)
    for index in indices:
        ax.add_patch(
            patches.Rectangle(
                (index[1] - target.shape[2] // 2, index[0] - target.shape[1] // 2),
                target.shape[2], target.shape[1],
                edgecolor='cyan',
                fill=False,
            ))

    plt.figure()
    plt.imshow(conv, cmap='jet')
    plt.colorbar()
    plt.show()


name = 'image1.png'
icon_name = 'target1.png'

# firstRgbResult(name, icon_name)

finalResult(name, icon_name)
