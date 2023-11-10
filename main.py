import matplotlib.colors
from PIL import Image
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np


def my_padding(img, shape, boundary=0):
    row, col = len(img), len(img[0])
    pad_sizeY, pad_sizeX = shape[0] // 2, shape[1] // 2
    res = np.zeros((row + (2 * pad_sizeY), col + (2 * pad_sizeX)), dtype=np.float_)
    pad_row, pad_col = len(res), len(res[0])
    if pad_sizeY == 0:
        res[pad_sizeY:, pad_sizeX:-pad_sizeX] = img.copy()
    elif pad_sizeX == 0:
        res[pad_sizeY:-pad_sizeY, pad_sizeX:] = img.copy()
    else:
        res[pad_sizeY:-pad_sizeY, pad_sizeX:-pad_sizeX] = img.copy()
    if boundary == 0:
        return res
    elif boundary == 1:
        res[0:pad_sizeY, 0:pad_sizeX] = img[0, 0]
        res[-pad_sizeY:, 0:pad_sizeX] = img[row - 1, 0]
        res[0:pad_sizeY, -pad_sizeX:] = img[0, col - 1]
        res[-pad_sizeY:, -pad_sizeX:] = img[row - 1, col - 1]

        res[0:pad_sizeY, pad_sizeX:pad_col - pad_sizeX] = np.repeat(img[0:1, 0:], [pad_sizeY], axis=0)  # 상단
        res[pad_row - pad_sizeY:, pad_sizeX:pad_col - pad_sizeX] = np.repeat(img[row - 1:row, 0:], [pad_sizeY],
                                                                             axis=0)  # 하단
        res[pad_sizeY:pad_row - pad_sizeY, 0:pad_sizeX] = np.repeat(img[0:, 0:1], [pad_sizeX], axis=1)  # 좌측
        res[pad_sizeY:pad_row - pad_sizeY, pad_col - pad_sizeX:] = np.repeat(img[0:, col - 1:col], [pad_sizeX],
                                                                             axis=1)  # 우측
        return res
    else:
        res[0:pad_sizeY, 0:pad_sizeX] = np.flip(img[0:pad_sizeY, 0:pad_sizeX])  # 좌측 상단
        res[-pad_sizeY:, 0:pad_sizeX] = np.flip(img[-pad_sizeY:, 0:pad_sizeX])  # 좌측 하단
        res[0:pad_sizeY, -pad_sizeX:] = np.flip(img[0:pad_sizeY, -pad_sizeX:])  # 우측 상단
        res[-pad_sizeY:, -pad_sizeX:] = np.flip(img[-pad_sizeY:, -pad_sizeX:])  # 우측 하단

        res[pad_sizeY:pad_row - pad_sizeY, 0:pad_sizeX] = np.flip(img[0:, 0:pad_sizeX], 1)  # 좌측
        res[pad_sizeY:pad_row - pad_sizeY, pad_col - pad_sizeX:] = np.flip(img[0:, col - pad_sizeX:], 1)  # 우측
        res[0:pad_sizeY, pad_sizeX:pad_col - pad_sizeX] = np.flip(img[0:pad_sizeY, 0:], 0)  # 상단
        res[pad_row - pad_sizeY:, pad_sizeX:pad_col - pad_sizeX] = np.flip(img[row - pad_sizeY:, 0:], 0)  # 하단
        return res


def my_getGKernel(shape, sigma):
    a = shape[0]
    b = shape[1]
    y, x = np.ogrid[-b:b + 1, -a:a + 1]
    gaus_kernel = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    sum = gaus_kernel.sum()
    gaus_kernel /= sum
    return gaus_kernel


def my_filtering(img, kernel, boundary=0):
    row, col = len(img), len(img[0])
    ksizeY, ksizeX = kernel.shape[0], kernel.shape[1]

    pad_image = my_padding(img, (ksizeY, ksizeX), boundary=boundary)
    filtered_img = np.zeros((row, col), dtype=np.float32)

    for i in range(row):
        for j in range(col):
            filtered_img[i, j] = np.sum(
                np.multiply(kernel, pad_image[i:i + ksizeY, j:j + ksizeX]))

    return filtered_img


def get_pixel_data(name):
    gray = np.array(Image.open(name).convert("L"))
    h, w = gray.shape
    gaus1 = my_getGKernel((5, 5), 1.6)
    gaus2 = my_getGKernel((5, 5), 1)
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
    gray = np.array(Image.open(name).convert("L"))
    gray_flip = np.flip(gray)
    h, w = gray_flip.shape
    gaus1 = my_getGKernel((5, 5), 1.6)
    gaus2 = my_getGKernel((5, 5), 1)
    gaussian1 = my_filtering(gray_flip, gaus1)
    gaussian2 = my_filtering(gray_flip, gaus2)
    DoG = np.zeros_like(gray_flip)
    for i in range(h):
        for j in range(w):
            DoG[i][j] = float(gaussian1[i][j]) - float(gaussian2[i][j])
    result = np.array(DoG).reshape(1, h, w)
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    return result


def calc_convolution(X, filters, pad=1):
    _, h, w = X.shape
    filters = np.flip(filters)
    filter_c, filter_h, filter_w = filters.shape

    out_h = (h + 2 * pad - filter_h)
    out_w = (w + 2 * pad - filter_w)

    in_X = np.pad(X, [(0, 0), (pad, pad), (pad, pad)], 'constant')
    out = np.zeros((out_h, out_w))

    for c in range(filter_c):
        for h in range(out_h):
            h_start = h
            h_end = h_start + filter_h
            for w in range(out_w):
                w_start = w
                w_end = w_start + filter_w
                out[h, w] = np.sum(in_X[c, h_start:h_end, w_start:w_end] * filters[c])

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
                (index[1], index[0]),
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
