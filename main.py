from PIL import Image
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
import sys


def my_padding(image, shape):
    h, w = shape
    pad_h, pad_w = int(h / 2), int(w / 2)
    res = np.pad(image, [(pad_h, h - pad_h), (pad_w, w - pad_w)], 'constant')
    return res


def my_kernel(shape, sigma):
    a, b = shape
    y, x = np.ogrid[-b:b + 1, -a:a + 1]

    gaus_kernel = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    sum = gaus_kernel.sum()
    gaus_kernel /= sum

    return gaus_kernel


def my_filtering(image, kernel):
    h, w = image.shape
    kernel_h, kernel_w = kernel.shape

    padded = my_padding(image, (kernel_h, kernel_w))
    filtered = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            filtered[i, j] = np.sum(padded[i:i + kernel_h, j:j + kernel_w] * kernel)

    return filtered


def get_pixel_data(name):
    gray = np.array(Image.open(name).convert("L"))
    h, w = gray.shape

    kernel1 = my_kernel((5, 5), 1.6)
    kernel2 = my_kernel((5, 5), 1)

    filtered1 = my_filtering(gray, kernel1)
    filtered2 = my_filtering(gray, kernel2)

    dog = np.zeros_like(gray)
    for i in range(h):
        for j in range(w):
            dog[i][j] = filtered1[i][j] - filtered2[i][j]

    normalized = (dog - np.min(dog)) / (np.max(dog) - np.min(dog))

    return normalized


def get_pixel_data_target(name):
    result = get_pixel_data(name)
    result = np.flip(result)
    return result


def calc_convolution(image, filter):
    h, w = image.shape
    filter = np.flip(filter)
    filter_h, filter_w = filter.shape

    image = my_padding(image, (filter_h, filter_w))
    result = np.zeros((h, w))

    for now_h in range(h):
        start_h = now_h
        end_h = start_h + filter_h
        for now_w in range(w):
            start_w = now_w
            end_w = start_w + filter_w
            result[now_h, now_w] = np.sum(image[start_h:end_h, start_w:end_w] * filter)

    return result


def finalResult(image_name, target_name):
    origin = np.array(Image.open(image_name).convert("L"))
    image = get_pixel_data(image_name)
    target = get_pixel_data_target(target_name)

    conv = calc_convolution(image, target)

    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.show()

    fig, ax = plt.subplots()
    ax.imshow(origin, cmap='gray')
    indices = np.argwhere(conv >= np.max(conv) * .9)
    for index in indices:
        ax.add_patch(
            patches.Rectangle(
                (index[1] - target.shape[1] // 2, index[0] - target.shape[0] // 2),
                target.shape[1], target.shape[0],
                edgecolor='cyan',
                fill=False,
            ))

    plt.figure()
    plt.imshow(conv, cmap='jet')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('이미지 파일의 이름과 아이콘 파일의 이름을 입력해야 합니다.')
        print('Ex) python main.py image1.png target1.png')
    else:
        name = sys.argv[1]
        icon_name = sys.argv[2]

        finalResult(name, icon_name)
