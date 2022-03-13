import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


def preprocess(img_BGR_lane):
    """
    Converts image to grayscale, performs gaussian blur and canny edge detection
    :param img_BGR_lane: Canny edge detection image
    :return:
    """
    grayscale_lane = cv2.cvtColor(img_BGR_lane, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grayscale_lane, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    plt.imshow(canny)
    plt.show()

    return canny


def mask(image):
    """
    Isolates region of interest
    :param image: Canny edge-detected image
    :return: masked region of interest
    """
    height = image.shape[0]
    polygons = np.array([
        [(50, height - 100), (1200, height - 100), (650, 300)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

def hough_accumulator(image):
    """
    Creates hough accumulator
    :param image:
    :return:
    """
    height = image.shape[0]
    width = image.shape[1]
    diagonal = np.ceil(np.sqrt((height ** 2) + (width ** 2)))
    rho_values = np.arange(-diagonal, diagonal + 1, 1)
    theta_values = np.deg2rad(np.arange(-90, 90, 1))

    hough_accumulator = np.zeros((len(rho_values), len(theta_values)), dtype=np.uint64)
    y_indices, x_indices = np.nonzero(image)  # edge pixel indices

    # vote
    for i in range(0, len(x_indices)):
        x = x_indices[i]
        y = y_indices[i]

        for j in range(len(theta_values)):
            rho = int((x * np.cos(theta_values[j]) + y * np.sin(theta_values[j])) + diagonal)
            hough_accumulator[rho][j] += 1

    return hough_accumulator, rho_values, theta_values


def peaks(hough_accumulator):
    """
    returns line indices, while ignoring neighboring pixels surrounding maxima
    :param hough_accumulator:
    :return: accumulator, index array
    """
    indices = []
    peaks = 2  # number of lines
    neighborhood = 100  # ignored pixels
    h_y = hough_accumulator.shape[0]
    h_x = hough_accumulator.shape[1]

    for i in range(0, peaks):
        index = np.argmax(hough_accumulator)
        h_index = np.unravel_index(index, hough_accumulator.shape)
        indices.append(h_index)

        y_index, x_index = h_index
        # y values
        if (y_index - (neighborhood/2)) < 0:
            y_min = 0
        else:
            y_min = y_index - (neighborhood / 2)
        if (y_index + (neighborhood / 2) + 1) > h_y:
            y_max = h_y
        else:
            y_max = y_index + (neighborhood / 2) + 1

        # x values
        if (x_index - (neighborhood/2)) < 0:
            x_min = 0
        else:
            x_min = x_index - (neighborhood / 2)
        if (x_index + (neighborhood / 2) + 1) > h_x:
            x_max = h_x
        else:
            x_max = x_index + (neighborhood / 2) + 1

        # set current line to 0
        for y in range(int(y_min), int(y_max)):
            for x in range(int(x_min), int(x_max)):
                hough_accumulator[y][x] = 0

    return indices


def draw_lines(image, indices, rho_values, theta_values):
    """
    Draws
    :param image:
    :param indices:
    :param rho_values:
    :param theta_values:
    :return:
    """
    line_image = np.zeros_like(image)
    width = image.shape[1]

    for i in range(0, len(indices)):
        rho = rho_values[indices[i][0]]
        theta = theta_values[indices[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x_0 = a * rho
        y_0 = b * rho

        x1 = int(x_0 + 1500 * -b)
        y1 = int(y_0 + 1500 * a)
        x2 = int(x_0 - 1500 * -b)
        y2 = int(y_0 - 1500 * a)

        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    # trim lines
    for h in range(0, 300):
        for w in range(0, width):
            line_image[h][w] = 0

    cv2.imshow('lines', line_image)
    cv2.waitKey(0)

    return line_image


if __name__ == "__main__":
    print('===========================================================')
    print('PSU CS 410/510, Winter 2022, Final Project - Lane Detection')
    print('===========================================================')

    path_file_image_source = sys.argv[1]
    path_file_image_result = sys.argv[2]

    # ===== read input image
    img_BGR_source = cv2.imread(path_file_image_source)
    img_BGR_lane = np.copy(img_BGR_source)

    # ===== preprocess image
    canny = preprocess(img_BGR_lane)
    masked_image = mask(canny)
    cv2.imshow('mask', masked_image)
    cv2.waitKey(0)

    # ===== Hough Lines
    accumulator, rho_values, theta_values = hough_accumulator(masked_image)
    indices = peaks(accumulator)
    line_image = draw_lines(img_BGR_lane, indices, rho_values, theta_values)

    # ===== Combine lines with original image
    combo_image = cv2.addWeighted(img_BGR_lane, 0.8, line_image, 1, 1)
    cv2.imshow('result', combo_image)
    cv2.waitKey(0)

    # ===== Write output image
    cv2.imwrite(filename=path_file_image_result, img=combo_image.clip(0.0, 255.0).astype(np.uint8))
