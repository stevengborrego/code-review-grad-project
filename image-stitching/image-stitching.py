import sys
import random
import math
import cv2
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojection_error=3, max_num_trial=3000):
    '''
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points, transform the first set of feature point to the second (e.g. warp image 1 to image 2)
    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojection_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    '''
    best_H = None
    best_H_num_inliers = 0
    for i in range(max_num_trial):
        samples = random.sample(list_pairs_matched_keypoints, 4)  # samples
        points_1 = [item[0] for item in samples]
        points_2 = [item[1] for item in samples]

        # Compute homography
        index = 0
        P = np.zeros((8, 9), dtype=np.float32)

        for j in range(0, len(points_1)):
            x_0 = points_1[j][0]
            y_0 = points_1[j][1]

            x_1 = points_2[j][0]
            y_1 = points_2[j][1]

            P[index] = [-x_0, -y_0, -1, 0, 0, 0, x_0 * x_1, y_0 * x_1, x_1]
            P[index + 1] = [0, 0, 0, -x_0, -y_0, -1, x_0 * y_1, y_0 * y_1, y_1]

            index += 2

        U, S, V = np.linalg.svd(np.array(P, dtype=np.float32))

        H = V[-1::].reshape(3, 3) / V[-1, -1]

        # count inliers
        num_inliers = 0
        for (p1, p2) in list_pairs_matched_keypoints:
            p1h = np.transpose([p1[0], p1[1], 1])
            p2h = np.transpose([p2[0], p2[1], 1])
            point1_predicted = np.dot(H, p1h)
            point1_predicted_standard = point1_predicted / point1_predicted[2]
            error = np.linalg.norm(p2h - point1_predicted_standard)
            if error < threshold_reprojection_error:
                num_inliers += 1

        percent_inliers = num_inliers / len(list_pairs_matched_keypoints)
        if percent_inliers > threshold_ratio_inliers:
            print("Percent Inliers: " + str(percent_inliers))
            return H

        # In case we don't find an H above the threshold, return the next best one
        if num_inliers > best_H_num_inliers:
            best_H_num_inliers = num_inliers
            best_H = H

    print("Percent Inliers: " + str(best_H_num_inliers))
    return best_H


def ex_extract_and_match_feature(img_1, img_2, ratio_robustness=0.7):
    '''
    1/ extract SIFT feature from image 1 and image 2,
    2/ use a bruteforce search to find pairs of matched features: for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points
    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_robustness: ratio for the robustness test
    :return list_pairs_matched_keypoints: has the format as list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    '''
    # ==============================
    # ===== 1/ extract features from input image 1 and image 2
    # ==============================

    list_pairs_matched_keypoints = []

    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    img_1_keypoints, img_1_descriptors = sift.detectAndCompute(img_1_gray, None)
    img_2_keypoints, img_2_descriptors = sift.detectAndCompute(img_2_gray, None)

    # ==============================
    # ===== 2/ use bruteforce search to find a list of pairs of matched feature points
    # ==============================

    for i in range(len(img_1_keypoints)):
        temp_list = []
        for j in range(len(img_2_keypoints)):
            distance = np.linalg.norm(img_1_descriptors[i] - img_2_descriptors[j])
            temp_list.append((j, distance))
        temp_list.sort(key=lambda x: x[1])
        if temp_list[0][1] / temp_list[1][1] < ratio_robustness:
            p = img_1_keypoints[i]
            q = img_2_keypoints[temp_list[0][0]]
            p1x = p.pt[0]
            p1y = p.pt[1]
            p2x = q.pt[0]
            p2y = q.pt[1]
            list_pairs_matched_keypoints.append([[p1x, p1y], [p2x, p2y]])

    return list_pairs_matched_keypoints


def ex_warp_blend_crop_image(img_1, H_1, img_2):
    '''
    1/ warp image img_1 using the homography H_1 to align it with image img_2 (using backward warping and bilinear resampling)
    2/ stitch image img_1 to image img_2 and apply average blending to blend the 2 images into a single panorama image
    3/ find the best bounding box for the resulting stitched image
    :param img_1:
    :param H_1:
    :param img_2:
    :return img_panorama: resulting panorama image
    '''
    # =====  use a backward warping algorithm to warp the source
    # 1/ to do so, we first create the inverse transform; 2/ use bilinear interpolation for resampling

    mask_input = np.ones(img_1.shape[0:2], np.float32)
    h = img_1.shape[0]
    w = img_1.shape[1]
    inverse_H = np.linalg.inv(H_1)
    canvas_img1 = np.zeros((3 * h, 3 * w, 3), dtype=np.float32)
    mask_img1 = np.zeros((3 * h, 3 * w), dtype=np.float32)

    for y in range(-h, 2 * h):
        for x in range(-w, 2 * w):
            dst_coordinateH = np.array([x, y, 1.0], np.float32)
            src_coordinateH = np.dot(inverse_H, dst_coordinateH)
            src_coordinate_Standard = src_coordinateH / src_coordinateH[2]

            # check in range
            if (0 < src_coordinate_Standard[0] < (w - 1)) and (0 < src_coordinate_Standard[1] < (h - 1)):
                i = int(math.floor(src_coordinate_Standard[0]))
                j = int(math.floor(src_coordinate_Standard[1]))
                a = src_coordinate_Standard[0] - i
                b = src_coordinate_Standard[1] - j
                canvas_img1[y+h, x+w] = (1-a)*(1-b) * img_1[j, i] + a*(1-b) * img_1[j, i+1]
                mask_img1[y+h, x+w] = (1-a)*(1-b) * mask_input[j, i] + a*(1-b) * mask_input[j, i+1]

    # ===== blend images: average blending
    h = img_2.shape[0]
    w = img_2.shape[1]
    canvas_img2 = np.zeros((3 * h, 3 * w, 3), dtype=np.float32)
    mask_img2 = np.zeros((3 * h, 3 * w), dtype=np.float32)
    canvas_img2[h:h*2, w:w*2] = img_2
    mask_img2[h:h*2, w:w*2] = 1.0

    img = canvas_img1 + canvas_img2
    mask = mask_img1 + mask_img2
    mask = np.tile(np.expand_dims(mask, 2), (1, 1, 3))
    img = np.divide(img, mask)

    # ===== find the best bounding box for the resulting stitched image so that it will contain all pixels from 2 original images
    mask_check = 1.0-np.float32(mask[:, :, 0] > 0)
    check_h = np.sum(mask_check[:, :], 1)
    check_w = np.sum(mask_check[:, :], 0)
    bottom = np.min(np.where(check_h < w * 3))
    top = np.max(np.where(check_h < w * 3))
    left = np.min(np.where(check_w < h * 3))
    right = np.max(np.where(check_w < h * 3))

    img_panorama = img[bottom:top, left:right]
    return img_panorama


def stitch_images(img_1, img_2):
    '''
    :param img_1: input image 1. We warp this image to align and stich it to the image 2
    :param img_2: is the reference image. We will not warp this image
    :return img_panorama: the resulting stiched image
    '''
    print('==============================')
    print('===== stitch two images to generate one panorama image')
    print('==============================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = ex_extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_robustness=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 1 to align it to image 2
    H_1 = ex_find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85, threshold_reprojection_error=3, max_num_trial=1000)

    # ===== warp image 1, blend it with image 2 using average blending to produce the resulting panorama image
    img_panorama = ex_warp_blend_crop_image(img_1=img_1, H_1=H_1, img_2=img_2)

    return img_panorama


if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2022, HW2: image stitching')
    print('==================================================')

    path_file_image_1 = sys.argv[1]
    path_file_image_2 = sys.argv[2]
    path_file_image_result = sys.argv[3]

    # ===== read 2 input images
    img_1 = cv2.imread(path_file_image_1)
    img_2 = cv2.imread(path_file_image_2)

    # ===== create a panorama image by stitch image 1 to image 2
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    cv2.imwrite(filename=path_file_image_result, img=img_panorama.clip(0.0, 255.0).astype(np.uint8))
