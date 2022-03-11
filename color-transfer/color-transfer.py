import cv2
import numpy as np
import sys

RGB_to_LMS = np.array([[0.3811, 0.5783, 0.0402],
                       [0.1967, 0.7244, 0.0782],
                       [0.0241, 0.1288, 0.8444]])

LMS_to_RGB = np.array([[4.4679, -3.5873, 0.1193],
                       [-1.2186, 2.3809, -0.1624],
                       [0.0497, -0.2439, 1.2045]])

LMS_to_CIECAM97s = np.array([[2.00, 1.00, 0.095],
                             [1.00, -1.09, 0.09],
                             [0.11, 0.11, -0.22]])


def convert_color_space_BGR_to_RGB(img_BGR):
    img_RGB = img_BGR[:, :, ::-1]
    return img_RGB


def convert_color_space_RGB_to_BGR(img_RGB):
    img_BGR = img_RGB[:, :, ::-1]
    return img_BGR


def convert_color_space_RGB_to_Lab(img_RGB):
    '''
    convert image color space RGB to Lab
    '''
    height = img_RGB.shape[0]
    width = img_RGB.shape[1]

    # Convert to LMS Cone Space
    img_LMS = np.zeros_like(img_RGB, dtype=np.float32)

    for h in range(height):
        for w in range(width):
            rgb_pixel = img_RGB[h, w]
            img_LMS[h, w] = np.matmul(RGB_to_LMS, rgb_pixel)

    # Convert LMS to LMS-log space
    img_LMS = np.log10(img_LMS)

    # Convert LMS-Log space to LAB space
    img_Lab = np.zeros_like(img_RGB, dtype=np.float32)
    m1 = np.array([[0.57735026919, 0, 0],   # 1/sqrt(3)
                   [0, 0.40824829046, 0],   # 1/sqrt(6)
                   [0, 0, 0.70710678118]])  # 1/sqrt(2)
    m2 = np.array([[1, 1, 1],
                   [1, 1, -2],
                   [1, -1, 0]])

    LMS_to_LAB = np.matmul(m1, m2)

    for h in range(height):
        for w in range(width):
            lms_pixel = img_LMS[h, w]
            img_Lab[h, w] = np.matmul(LMS_to_LAB, lms_pixel)

    return img_Lab


def convert_color_space_Lab_to_RGB(img_Lab):
    '''
    convert image color space Lab to RGB
    '''
    height = img_Lab.shape[0]
    width = img_Lab.shape[1]

    img_LMS = np.zeros_like(img_Lab, dtype=np.float32)

    m1 = np.array([[1, 1, 1],
                   [1, 1, -1],
                   [1, -2, 0]])
    m2 = np.array([[0.57735026919, 0, 0],   # sqrt(3)/3
                   [0, 0.40824829046, 0],   # sqrt(6)/6
                   [0, 0, 0.70710678118]])  # 1/sqrt(2)

    LAB_to_LMS = np.matmul(m1, m2)

    for h in range(height):
        for w in range(width):
            lab_pixel = img_Lab[h, w]
            img_LMS[h, w] = np.matmul(LAB_to_LMS, lab_pixel)

    # undo log10
    img_LMS = pow(10, img_LMS)

    img_RGB = np.zeros_like(img_Lab, dtype=np.float32)

    for h in range(height):
        for w in range(width):
            lms_pixel = img_LMS[h, w]
            img_RGB[h, w] = np.matmul(LMS_to_RGB, lms_pixel)

    return img_RGB


def convert_color_space_RGB_to_CIECAM97s(img_RGB):
    '''
    convert image color space RGB to CIECAM97s
    '''
    height = img_RGB.shape[0]
    width = img_RGB.shape[1]
    img_CIECAM97s = np.zeros_like(img_RGB, dtype=np.float32)

    # Convert to LMS Cone Space
    img_LMS = np.zeros_like(img_RGB, dtype=np.float32)

    for h in range(height):
        for w in range(width):
            rgb_pixel = img_RGB[h, w]
            img_LMS[h, w] = np.matmul(RGB_to_LMS, rgb_pixel)

    for h in range(height):
        for w in range(width):
            lms_pixel = img_LMS[h, w]
            img_CIECAM97s[h, w] = np.matmul(LMS_to_CIECAM97s, lms_pixel)

    return img_CIECAM97s


def convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s):
    '''
    convert image color space CIECAM97s to RGB
    '''
    height = img_CIECAM97s.shape[0]
    width = img_CIECAM97s.shape[1]

    img_RGB = np.zeros_like(img_CIECAM97s, dtype=np.float32)
    img_LMS = np.zeros_like(img_RGB, dtype=np.float32)

    LMS_to_RGB = np.array([[4.4679, -3.5873, 0.1193],
                           [-1.2186, 2.3809, -0.1624],
                           [0.0497, -0.2439, 1.2045]])

    LMS_to_CIECAM97s = np.array([[2.00, 1.00, 0.095],
                                 [1.00, -1.09, 0.09],
                                 [0.11, 0.11, - 0.22]])

    CIECAM97s_to_LMS = np.linalg.inv(LMS_to_CIECAM97s)

    for h in range(height):
        for w in range(width):
            ciecam97s_pixel = img_CIECAM97s[h, w]
            img_LMS[h, w] = np.matmul(CIECAM97s_to_LMS, ciecam97s_pixel)

    for h in range(height):
        for w in range(width):
            lms_pixel = img_LMS[h, w]
            img_RGB[h, w] = np.matmul(LMS_to_RGB, lms_pixel)

    return img_RGB


def color_transfer_in_Lab(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_Lab =====')
    height = img_RGB_source.shape[0]
    width = img_RGB_source.shape[1]

    # convert to LAB space
    img_LAB_source = convert_color_space_RGB_to_Lab(img_RGB_source)
    img_LAB_target = convert_color_space_RGB_to_Lab(img_RGB_target)

    # source channels
    s_l = np.array(img_LAB_source[:, :, 0])
    s_a = np.array(img_LAB_source[:, :, 1])
    s_b = np.array(img_LAB_source[:, :, 2])

    # target channels
    t_l = np.array(img_LAB_target[:, :, 0])
    t_a = np.array(img_LAB_target[:, :, 1])
    t_b = np.array(img_LAB_target[:, :, 2])

    # compute mean
    s_l_mean = np.mean(s_l)
    s_a_mean = np.mean(s_a)
    s_b_mean = np.mean(s_b)

    t_l_mean = np.mean(t_l)
    t_a_mean = np.mean(t_a)
    t_b_mean = np.mean(t_b)

    # subtract mean
    for h in range(height):
        for w in range(width):
            img_LAB_source[h, w, 0] = img_LAB_source[h, w, 0] - s_l_mean
            img_LAB_source[h, w, 1] = img_LAB_source[h, w, 1] - s_a_mean
            img_LAB_source[h, w, 2] = img_LAB_source[h, w, 2] - s_b_mean

    # compute standard deviation
    t_l_stddev = np.std(t_l)
    t_a_stddev = np.std(t_a)
    t_b_stddev = np.std(t_b)

    s_l_stddev = np.std(s_l)
    s_a_stddev = np.std(s_a)
    s_b_stddev = np.std(s_b)

    # transfer color to source image
    for h in range(height):
        for w in range(width):
            img_LAB_source[h, w, 0] = ((t_l_stddev / s_l_stddev) * img_LAB_source[h, w, 0]) + t_l_mean
            img_LAB_source[h, w, 1] = ((t_a_stddev / s_a_stddev) * img_LAB_source[h, w, 1]) + t_a_mean
            img_LAB_source[h, w, 2] = ((t_b_stddev / s_b_stddev) * img_LAB_source[h, w, 2]) + t_b_mean

    # convert back to RGB
    img_RGB_new = convert_color_space_Lab_to_RGB(img_LAB_source)

    return img_RGB_new


def color_transfer_in_RGB(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_RGB =====')
    height = img_RGB_source.shape[0]
    width = img_RGB_source.shape[1]

    img_RGB_source_float = np.zeros_like(img_RGB_source, dtype=np.float32)

    # source channels
    s_r = np.array(img_RGB_source[:, :, 0])
    s_g = np.array(img_RGB_source[:, :, 1])
    s_b = np.array(img_RGB_source[:, :, 2])

    # target channels
    t_r = np.array(img_RGB_target[:, :, 0])
    t_g = np.array(img_RGB_target[:, :, 1])
    t_b = np.array(img_RGB_target[:, :, 2])

    # compute mean
    s_r_mean = np.mean(s_r)
    s_g_mean = np.mean(s_g)
    s_b_mean = np.mean(s_b)

    t_r_mean = np.mean(t_r)
    t_g_mean = np.mean(t_g)
    t_b_mean = np.mean(t_b)

    # subtract mean
    for h in range(height):
        for w in range(width):
            img_RGB_source_float[h, w, 0] = img_RGB_source[h, w, 0] - s_r_mean
            img_RGB_source_float[h, w, 1] = img_RGB_source[h, w, 1] - s_g_mean
            img_RGB_source_float[h, w, 2] = img_RGB_source[h, w, 2] - s_b_mean

    # compute standard deviation
    t_r_stddev = np.std(t_r)
    t_g_stddev = np.std(t_g)
    t_b_stddev = np.std(t_b)

    s_r_stddev = np.std(s_r)
    s_g_stddev = np.std(s_g)
    s_b_stddev = np.std(s_b)

    # transfer color to source image
    for h in range(height):
        for w in range(width):
            img_RGB_source_float[h, w, 0] = ((t_r_stddev / s_r_stddev) * img_RGB_source_float[h, w, 0]) + t_r_mean
            img_RGB_source_float[h, w, 1] = ((t_g_stddev / s_g_stddev) * img_RGB_source_float[h, w, 1]) + t_g_mean
            img_RGB_source_float[h, w, 2] = ((t_b_stddev / s_b_stddev) * img_RGB_source_float[h, w, 2]) + t_b_mean

    return img_RGB_source_float


def color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_CIECAM97s =====')
    height = img_RGB_source.shape[0]
    width = img_RGB_source.shape[1]

    # convert to LAB space
    img_CIECAM97s_source = convert_color_space_RGB_to_CIECAM97s(img_RGB_source)
    img_CIECAM97s_target = convert_color_space_RGB_to_CIECAM97s(img_RGB_target)

    # source channels
    s_A = np.array(img_CIECAM97s_source[:, :, 0])
    s_C1 = np.array(img_CIECAM97s_source[:, :, 1])
    s_C2 = np.array(img_CIECAM97s_source[:, :, 2])

    # target channels
    t_A = np.array(img_CIECAM97s_target[:, :, 0])
    t_C1 = np.array(img_CIECAM97s_target[:, :, 1])
    t_C2 = np.array(img_CIECAM97s_target[:, :, 2])

    # compute mean
    s_A_mean = np.mean(s_A)
    s_C1_mean = np.mean(s_C1)
    s_C2_mean = np.mean(s_C2)

    t_A_mean = np.mean(t_A)
    t_C1_mean = np.mean(t_C1)
    t_C2_mean = np.mean(t_C2)

    # subtract mean
    for h in range(height):
        for w in range(width):
            img_CIECAM97s_source[h, w, 0] = img_CIECAM97s_source[h, w, 0] - s_A_mean
            img_CIECAM97s_source[h, w, 1] = img_CIECAM97s_source[h, w, 1] - s_C1_mean
            img_CIECAM97s_source[h, w, 2] = img_CIECAM97s_source[h, w, 2] - s_C2_mean

    # compute standard deviation
    t_A_stddev = np.std(t_A)
    t_C1_stddev = np.std(t_C1)
    t_C2_stddev = np.std(t_C2)

    s_A_stddev = np.std(s_A)
    s_C1_stddev = np.std(s_C1)
    s_C2_stddev = np.std(s_C2)

    # transfer color to source image
    for h in range(height):
        for w in range(width):
            img_CIECAM97s_source[h, w, 0] = ((t_A_stddev / s_A_stddev) * img_CIECAM97s_source[h, w, 0]) + t_A_mean
            img_CIECAM97s_source[h, w, 1] = ((t_C1_stddev / s_C1_stddev) * img_CIECAM97s_source[h, w, 1]) + t_C1_mean
            img_CIECAM97s_source[h, w, 2] = ((t_C2_stddev / s_C2_stddev) * img_CIECAM97s_source[h, w, 2]) + t_C2_mean

    # convert back to RGB
    img_RGB_new = convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s_source)

    return img_RGB_new


def color_transfer(img_RGB_source, img_RGB_target, option):
    if option == 'in_RGB':
        img_RGB_new = color_transfer_in_RGB(img_RGB_source, img_RGB_target)
    elif option == 'in_Lab':
        img_RGB_new = color_transfer_in_Lab(img_RGB_source, img_RGB_target)
    elif option == 'in_CIECAM97s':
        img_RGB_new = color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target)
    return img_RGB_new


if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2020, HW1: color transfer')
    print('==================================================')

    path_file_image_source = sys.argv[1]
    path_file_image_target = sys.argv[2]
    path_file_image_result_in_Lab = sys.argv[3]
    path_file_image_result_in_RGB = sys.argv[4]
    path_file_image_result_in_CIECAM97s = sys.argv[5]

    # ===== read input images
    # img_RGB_source: is the image you want to change the its color
    # img_RGB_target: is the image containing the color distribution that you want to change the img_RGB_source to
    # (transfer color of the img_RGB_target to the img_RGB_source)

    img_BGR_source = cv2.imread(path_file_image_source)
    img_BGR_target = cv2.imread(path_file_image_target)

    img_RGB_source = convert_color_space_BGR_to_RGB(img_BGR_source)
    img_RGB_target = convert_color_space_BGR_to_RGB(img_BGR_target)

    # ===== Color Transfer in LAB
    img_RGB_new_Lab = color_transfer(img_RGB_source, img_RGB_target, option='in_Lab')

    # Adjust pixel values
    img_BGR_new_Lab = convert_color_space_RGB_to_BGR(img_RGB_new_Lab)
    img_BGR_new_Lab = np.clip(img_BGR_new_Lab, 0, 255)
    img_BGR_new_Lab = np.uint8(img_BGR_new_Lab)

    # write to a file
    cv2.imwrite(path_file_image_result_in_Lab, img_BGR_new_Lab)

    # ===== Color Transfer in RGB
    img_RGB_new_RGB = color_transfer(img_RGB_source, img_RGB_target, option='in_RGB')
    # Adjust pixel values
    img_BGR_new_RGB = convert_color_space_RGB_to_BGR(img_RGB_new_RGB)
    img_BGR_new_RGB = np.clip(img_BGR_new_RGB, 0, 255)
    img_BGR_new_RGB = np.uint8(img_BGR_new_RGB)

    # write to a file
    cv2.imwrite(path_file_image_result_in_RGB, img_BGR_new_RGB)

    # ===== Color Transfer in CIECAM97s
    img_RGB_new_CIECAM97s = color_transfer(img_RGB_source, img_RGB_target, option='in_CIECAM97s')

    # Adjust pixel values
    img_BGR_new_CIECAM97s = convert_color_space_RGB_to_BGR(img_RGB_new_CIECAM97s)
    img_BGR_new_CIECAM97s = np.clip(img_BGR_new_CIECAM97s, 0, 255)
    img_BGR_new_CIECAM97s = np.uint8(img_BGR_new_CIECAM97s)

    cv2.imwrite(path_file_image_result_in_CIECAM97s, img_BGR_new_CIECAM97s)
    