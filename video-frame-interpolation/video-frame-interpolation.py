import cv2
import sys
import pickle
import numpy as np
import os

BLUR_OCC = 3


def linInterp(im, x, y):
    '''
    bilinear interpolation to get pixel value of a pixel [x,y] in image im
    :return pixel: pixel value
    :return inbounds: 1=in boundary, 0=out of boundary
    '''
    sh = im.shape
    w = sh[1]
    h = sh[0]
    inBounds = 1
    if x < 0:
        inBounds = 0
        x = 0.0
    elif x >= (w-1):
        inBounds = 0
        x = w-1.000001
    if y < 0:
        inBounds = 0
        y = 0.0
    elif y >= (h-1):
        inBounds = 0
        y = h-1.000001

    x0 = int(x)
    y0 = int(y)
    if x0 < (w-1):
        x1 = x0+1.0
    else:
        x1 = x0
    if y0 < (h-1):
        y1 = y0+1.0
    else:
        y1 = y0
    dx = np.float32(x) - np.float32(x0)
    dy = np.float32(y) - np.float32(y0)

    p00 = im[np.int32(y0), np.int32(x0)].astype(np.float32)
    p10 = im[np.int32(y0), np.int32(x1)].astype(np.float32)
    p01 = im[np.int32(y1), np.int32(x0)].astype(np.float32)
    p11 = im[np.int32(y1), np.int32(x1)].astype(np.float32)

    pixel = (1.0 - dx) * (1.0 - dy) * p00 + \
            dx * (1.0 - dy) * p10 + \
            (1.0 - dx) * dy * p01 + \
            dx * dy * p11

    return pixel, inBounds


def readFlowFile(file):
    '''
    credit: this function code is obtained from: https://github.com/Johswald/flow-code-python
    '''
    TAG_FLOAT = 202021.25
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file, 'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)[0]
    h = np.fromfile(f, np.int32, count=1)[0]
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()

    return flow


def find_holes(flow):
    '''
    Find a mask of holes in a given flow matrix
    Determine it is a hole if a vector length is too long: >10^9, of it contains NAN, of INF
    :param flow: an dense optical flow matrix of shape [h,w,2], containing a vector [ux,uy] for each pixel
    :return: a mask annotated 0=hole, 1=no hole
    '''
    height = flow.shape[0]
    width = flow.shape[1]
    holes = np.ones((height, width))

    for h in range(height):
        for w in range(width):
            f = flow[h, w]
            magnitude = np.linalg.norm(f)
            if (magnitude > 10**9) or np.isnan(magnitude) or np.isinf(magnitude):
                holes[h, w] = 0

    return holes


def holefill(flow, holes):
    '''
    fill holes in order: row then column, until fill in all the holes in the flow
    :param flow: matrix of dense optical flow, it has shape [h,w,2]
    :param holes: a binary mask that annotate the location of a hole, 0=hole, 1=no hole
    :return: flow: updated flow
    '''
    height, width, _ = flow.shape
    has_hole = 1

    while has_hole == 1:
        for h in range(height):
            for w in range(width):
                if holes[h, w] == 0:
                    arr = []  # neighbor pixel indices
                    arr.append([h-1, w-1])  # upper left
                    arr.append([h-1, w])    # upper
                    arr.append([h-1, w+1])  # upper right
                    arr.append([h, w-1])    # left
                    arr.append([h, w+1])    # right
                    arr.append([h+1, w-1])  # lower left
                    arr.append([h+1, w])    # lower
                    arr.append([h+1, w+1])  # lower right
                    sum_x = 0  # first flow component (x)
                    sum_y = 0  # second flow component (y)
                    count = 0  # number of non-hole neighbor pixels

                    for i in range(len(arr)):
                        # check if neighbor is a hole or out of bounds
                        y = arr[i][0]
                        x = arr[i][1]
                        if y not in range(height):
                            continue
                        if x not in range(width):
                            continue
                        if holes[y][x] == 0:
                            continue # skip pixel
                        sum_x += flow[y][x][0] # x
                        sum_y += flow[y][x][1] # y
                        count += 1

                    # fill hole in flow matrix
                    if count > 0:
                        flow[h][w][0] = np.float64(sum_x / count)
                        flow[h][w][1] = np.float64(sum_y / count)
                        holes[h][w] = 1 # update hole mask

        # check if there are still holes
        hole_flag = 0
        for h in range(height):
            for w in range(width):
                if holes[h][w] == 0:
                    hole_flag = 1
                    break
            if hole_flag == 1:
                break

        # no holes, so break while loop
        if hole_flag == 0:
            has_hole = 0

    return flow


def occlusions(flow0, frame0, frame1):
    '''
    Follow the step 3 in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
    for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.
    :param flow0: dense optical flow
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :return:
    '''
    height, width, _ = flow0.shape
    occ0 = np.zeros([height, width], dtype=np.float32)
    occ1 = np.zeros([height, width], dtype=np.float32)

    # ==================================================
    # ===== step 4/ warp flow field to target frame
    # ==================================================
    flow1 = interpflow(flow0, frame0, frame1, 1.0)
    pickle.dump(flow1, open('flow1.step4.data', 'wb'))
    # ====== score
    flow1       = pickle.load(open('flow1.step4.data', 'rb'))
    flow1_step4 = pickle.load(open('flow1.step4.sample', 'rb'))
    diff = np.sum(np.abs(flow1 - flow1_step4))
    print('flow1_step4', diff)

    # ==================================================
    # ===== main part of step 5
    # ==================================================

    for y in range(height):
        for x in range(width):
            fx = np.float32(flow0[y][x][0])
            fy = np.float32(flow0[y][x][1])

            assert(unknown_flow(fx, fy) == False)

            x1 = np.int32(x + fx + 0.5)
            y1 = np.int32(y + fy + 0.5)

            if 0 <= x1 < width and 0 <= y1 < height:
                fx1 = np.float32(flow1[y1][x1][0])
                fy1 = np.float32(flow1[y1][x1][1])
                diff_x = abs(fx1 - fx)
                diff_y = abs(fy1 - fy)
                abs_diff = abs(diff_x + diff_y)

                if abs_diff > 0.5:
                    occ0[y][x] = 1
            else:
                occ0[y][x] = 1

            fx = np.float32(flow1[y][x][0])
            fy = np.float32(flow1[y][x][1])

            if unknown_flow(fx, fy):
                occ1[y][x] = 1

    return occ0, occ1


def unknown_flow(fx, fy):
    '''
    determine if a flow vector is unknown
    :param fx: component x of a flow vector
    :param fy: component y of a flow vector
    :return:
    '''
    if (fx > np.power(10, 9)) or (fy > np.power(10, 9)) or np.isnan(fx) or np.isnan(fy):
        return True
    return False


def bilinear(src_img, src_x, src_y):
    # i corresponds to x, j corresponds to y
    height = src_img.shape[0]  # j
    width = src_img.shape[1]  # i
    i = int(np.floor(src_x))
    j = int(np.floor(src_y))
    a = src_x - i
    b = src_y - j
    # Both are past inner boundaries
    if i < 0 and j < 0:
        return src_img[0, 0]
    # Both are at (or past) outer boundaries
    if i >= width - 1 and j >= height - 1:
        return src_img[height - 1, width - 1]
    # width is past outer boundary, height is past inner boundary
    if i >= width - 1 and j < 0:
        return src_img[0, width - 1]
    # width is past inner boundary, height is past outer boundary
    if i < 0 and j >= height -1:
        return src_img[height - 1, 0]
    # width is at (or past) outer boundary
    if i >= width - 1:
        return np.float64(
                (1 - b) * src_img[j, width - 1]
                + b * src_img[j + 1, width - 1]
        )
    # width is past inner boundary
    if i < 0:
        return np.float64(
                (1 - b) * src_img[j, 0]
                + b * src_img[j + 1, 0]
        )
    # height is at (or past) outer boundary
    if j >= height -1:
        return np.float64(
            (1 - a) * src_img[height - 1, i]
            + a * src_img[height -1, i + 1]
        )
    # height is past inner boundary
    if j < 0:
        return np.float64(
                (1 - a) * src_img[0 - 1, i]
                + a * src_img[0, i + 1]
        )

    # Fall through to regular equation
    return np.float64(
        (1 - a) * (1 - b) * src_img[j, i]
        + a * (1 - b) * src_img[j + 1, i]
        + (a * b) * src_img[j + 1, i + 1]
        + ((1 - a) * b) * src_img[j, i + 1]
    )


def interpflow(flow, frame0, frame1, t):
    '''
    Forward warping flow (from frame0 to frame1) to a position t in the middle of the 2 frames
    Follow the algorithm (1) described in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
    for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

    :param flow: dense optical flow from frame0 to frame1
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :param t: the intermiddite position in the middle of the 2 input frames
    :return: a warped flow
    '''
    iflow = np.ones_like(flow, dtype=np.float32) * np.power(10, 10).astype(np.float32)

    height = frame0.shape[0]
    width = frame0.shape[1]

    colorDiffIm = np.ones([height, width], dtype=np.float32) * np.power(10, 10)

    for y in range(0, height):
        for x in range(0, width):
            fx = flow[y, x, 0]
            fy = flow[y, x, 1]
            if unknown_flow(fx=fx, fy=fy) == True:
                continue
            p0 = frame0[y, x]
            for yy in np.arange(-0.5, 0.51, 0.5):
                for xx in np.arange(-0.5, 0.51, 0.5):
                    p1, _ = linInterp(frame1.astype(np.float32), x + xx + fx, y + yy + fy)
                    colordiff = np.sum(np.abs(p0 - p1))
                    nx = np.int32(x + xx + t * fx + 0.5)
                    ny = np.int32(y + yy + t * fy + 0.5)
                    if (nx >= 0) and (nx < width) and (ny >= 0) and (ny < height):
                        if colordiff < colorDiffIm[ny, nx]:
                            iflow[ny, nx, 0] = fx
                            iflow[ny, nx, 1] = fy
                            colorDiffIm[ny, nx] = colordiff
    return iflow


def warpimages(iflow, frame0, frame1, occ0, occ1, t):
    '''
    Compute the colors of the interpolated pixels by inverse-warping frame 0 and frame 1 to the postion t based on the
    forwarded-warped flow iflow at t
    Follow the algorithm (4) described in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
     for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

    :param iflow: forwarded-warped (from flow0) at position t
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :param occ0: occlusion mask of frame 0
    :param occ1: occlusion mask of frame 1
    :param t: interpolated position t
    :return: interpolated image at position t in the middle of the 2 input frames
    '''

    iframe = np.zeros_like(frame0).astype(np.float32)
    height = frame0.shape[0]
    width = frame0.shape[1]

    t0 = t
    t1 = 1.0 - t

    for y in range(height):
        for x in range(width):
            fx = iflow[y][x][0].astype(np.float32)
            fy = iflow[y][x][1].astype(np.float32)

            if unknown_flow(fx, fy) == True:
                iframe[y][x][:] = 0
                continue

            x0 = np.float32(x) - t0 * fx
            y0 = np.float32(y) - t0 * fy
            x1 = np.float32(x) - t1 * fx
            y1 = np.float32(y) - t1 * fy

            p0, inbounds0 = linInterp(frame0, x0, y0)
            p1, inbounds1 = linInterp(frame1, x1, y1)

            o0, _ = linInterp(occ0, x0, y0)
            o1, _ = linInterp(occ1, x1, y1)

            oc0 = np.around(o0 + 1) - 1
            oc1 = np.around(o1 + 1) - 1

            if inbounds0 and inbounds1 and oc0 == 0 and oc1 == 0:
                w0 = t1
                w1 = t0
            else:
                if oc0 > oc1:
                    w0 = 0
                    w1 = 1
                else:
                    w1 = 0
                    w0 = 1

            iframe[y][x][:] = np.int32(w0 * p0 + w1 * p1 + 0.5)

    return iframe


def blur(im):
    '''
    blur using a gaussian kernel [5,5] using opencv function: cv2.GaussianBlur, sigma=0
    :param im:
    :return updated im:
    '''

    return cv2.GaussianBlur(im, (5, 5), sigmaX=0, sigmaY=0)


def internp(frame0, frame1, t=0.5, flow0=None):
    '''
    :param frame0: beggining frame
    :param frame1: ending frame
    :param flow0:
    :return frame_t: an interpolated frame at time t
    '''
    print('==============================')
    print('===== interpolate an intermediate frame at t=', str(t))
    print('==============================')

    # ==================================================
    # ===== 1/ find the optical flow between the two given images: from frame0 to frame1,
    #  if there is no given flow0, run opencv function to extract it
    # ==================================================
    if flow0 is None:
        i1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        i2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        flow0 = cv2.calcOpticalFlowFarneback(i1, i2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # ==================================================
    # ===== 2/ find holes in the flow
    # ==================================================
    holes0 = find_holes(flow0)
    pickle.dump(holes0, open('holes0.step2.data', 'wb'))  # save your intermediate result
    # ====== score
    holes0       = pickle.load(open('holes0.step2.data', 'rb')) # load your intermediate result
    holes0_step2 = pickle.load(open('holes0.step2.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(holes0 - holes0_step2))
    print('holes0_step2', diff)

    # ==================================================
    # ===== 3/ fill in any hole using an outside-in strategy
    # ==================================================
    flow0 = holefill(flow0, holes0)
    pickle.dump(flow0, open('flow0.step3.data', 'wb'))  # save your intermediate result
    # ====== score
    flow0       = pickle.load(open('flow0.step3.data', 'rb'))  # load your intermediate result
    flow0_step3 = pickle.load(open('flow0.step3.sample', 'rb'))  # load sample result
    diff = np.sum(np.abs(flow0 - flow0_step3))
    print('flow0_step3', diff)

    # ==================================================
    # ===== 5/ estimate occlusion mask
    # ==================================================
    occ0, occ1 = occlusions(flow0, frame0, frame1)
    pickle.dump(occ0, open('occ0.step5.data', 'wb'))  # save your intermediate result
    pickle.dump(occ1, open('occ1.step5.data', 'wb'))  # save your intermediate result
    # ===== score
    occ0        = pickle.load(open('occ0.step5.data', 'rb'))  # load your intermediate result
    occ1        = pickle.load(open('occ1.step5.data', 'rb'))  # load your intermediate result
    occ0_step5  = pickle.load(open('occ0.step5.sample', 'rb'))  # load sample result
    occ1_step5  = pickle.load(open('occ1.step5.sample', 'rb'))  # load sample result
    diff = np.sum(np.abs(occ0_step5 - occ0))
    print('occ0_step5', diff)
    diff = np.sum(np.abs(occ1_step5 - occ1))
    print('occ1_step5', diff)

    # ==================================================
    # ===== step 6/ blur occlusion mask
    # ==================================================
    for iblur in range(0, BLUR_OCC):
        occ0 = blur(occ0)
        occ1 = blur(occ1)
    pickle.dump(occ0, open('occ0.step6.data', 'wb'))  # save your intermediate result
    pickle.dump(occ1, open('occ1.step6.data', 'wb'))  # save your intermediate result
    # ===== score
    occ0        = pickle.load(open('occ0.step6.data', 'rb'))  # load your intermediate result
    occ1        = pickle.load(open('occ1.step6.data', 'rb'))  # load your intermediate result
    occ0_step6  = pickle.load(open('occ0.step6.sample', 'rb'))  # load sample result
    occ1_step6  = pickle.load(open('occ1.step6.sample', 'rb'))  # load sample result
    diff = np.sum(np.abs(occ0_step6 - occ0))
    print('occ0_step6', diff)
    diff = np.sum(np.abs(occ1_step6 - occ1))
    print('occ1_step6', diff)

    # ==================================================
    # ===== step 7/ forward-warp the flow to time t to get flow_t
    # ==================================================
    flow_t = interpflow(flow0, frame0, frame1, t)
    pickle.dump(flow_t, open('flow_t.step7.data', 'wb'))  # save your intermediate result
    # ====== score
    flow_t       = pickle.load(open('flow_t.step7.data', 'rb'))  # load your intermediate result
    flow_t_step7 = pickle.load(open('flow_t.step7.sample', 'rb'))  # load sample result
    diff = np.sum(np.abs(flow_t-flow_t_step7))
    print('flow_t_step7', diff)

    # ==================================================
    # ===== step 8/ find holes in the estimated flow_t
    # ==================================================
    holes1 = find_holes(flow_t)
    pickle.dump(holes1, open('holes1.step8.data', 'wb'))  # save your intermediate result
    # ====== score
    holes1       = pickle.load(open('holes1.step8.data', 'rb'))  # load your intermediate result
    holes1_step8 = pickle.load(open('holes1.step8.sample', 'rb'))  # load sample result
    diff = np.sum(np.abs(holes1-holes1_step8))
    print('holes1_step8', diff)

    # ===== fill in any hole in flow_t using an outside-in strategy
    flow_t = holefill(flow_t, holes1)
    pickle.dump(flow_t, open('flow_t.step8.data', 'wb'))  # save your intermediate result
    # ====== score
    flow_t       = pickle.load(open('flow_t.step8.data', 'rb'))  # load your intermediate result
    flow_t_step8 = pickle.load(open('flow_t.step8.sample', 'rb'))  # load sample result
    diff = np.sum(np.abs(flow_t-flow_t_step8))
    print('flow_t_step8', diff)

    # ==================================================
    # ===== 9/ inverse-warp frame 0 and frame 1 to the target time t
    # ==================================================
    frame_t = warpimages(flow_t, frame0, frame1, occ0, occ1, t)
    pickle.dump(frame_t, open('frame_t.step9.data', 'wb'))  # save your intermediate result
    # ====== score
    frame_t       = pickle.load(open('frame_t.step9.data', 'rb'))  # load your intermediate result
    frame_t_step9 = pickle.load(open('frame_t.step9.sample', 'rb'))  # load sample result
    diff = np.sqrt(np.mean(np.square(frame_t.astype(np.float32) - frame_t_step9.astype(np.float32))))
    print('frame_t', diff)

    return frame_t


if __name__ == "__main__":

    print('==================================================')
    print('PSU CS 410/510, Winter 2022, HW3: video frame interpolation')
    print('==================================================')

    # ===================================
    # example:
    # python interp_skeleton.py frame0.png frame1.png flow0.flo frame05.png
    # ===================================
    path_file_image_0 = sys.argv[1]
    path_file_image_1 = sys.argv[2]
    path_file_flow    = sys.argv[3]
    path_file_image_result = sys.argv[4]

    # ===== read 2 input images and flow
    frame0 = cv2.imread(path_file_image_0)
    frame1 = cv2.imread(path_file_image_1)
    flow0  = readFlowFile(path_file_flow)

    # ===== interpolate an intermediate frame at t, t in [0,1]
    frame_t = internp(frame0=frame0, frame1=frame1, t=0.5, flow0=flow0)
    cv2.imwrite(filename=path_file_image_result, img=(frame_t * 1.0).clip(0.0, 255.0).astype(np.uint8))
