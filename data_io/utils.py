import cv2
import numpy as np

NNSIZE = (256, 256, 3)

def scaleit3(img, size=None):
    if size is not None:
        NNSIZE = size
    imsz = img.shape
    rd = float(NNSIZE[0]) / float(np.max(imsz))
    mxdim = np.max(imsz)

    # FILL IT
    imgcanvas = np.zeros((NNSIZE[0], NNSIZE[1], NNSIZE[2]), dtype='uint8')
    offs_col = (mxdim - imsz[1]) / 2
    offs_row = (mxdim - imsz[0]) / 2

    offs_col = int(offs_col)
    offs_row = int(offs_row)

    # take cols
    imgcanvas = np.zeros((mxdim, mxdim, NNSIZE[2]), dtype='uint8')
    imgcanvas[offs_row:offs_row + imsz[0],
    offs_col:offs_col + imsz[1]] = img.reshape((imsz[0], imsz[1], NNSIZE[2]))

    # exit(1)
    # take rows
    if (offs_row):
        tr = img[0, :]
        br = img[-1, :]
        # print tr.shape , numpy.tile(tr, (offs_row,1,1)).shape
        # exit(1)
        imgcanvas[0:offs_row, :, :] = np.tile(tr, (offs_row, 1, 1))
        imgcanvas[-offs_row - 1:, :, :] = np.tile(br, (offs_row + 1, 1, 1))

    # take cols
    if (offs_col):
        # cv2.imshow("t", imgcanvas)
        # cv2.waitKey(0)
        lc = img[:, 0, :]
        rc = img[:, -1, :]
        # print lc.shape , numpy.tile(lc, (offs_col, 1, 1)).shape
        imgcanvas[:, 0:offs_col, :] = np.tile(lc,
                                                 (offs_col, 1, 1)).transpose(
            (1, 0, 2))
        # print imgcanvas.shape
        imgcanvas[:, -offs_col - 1:, :] = np.tile(rc, (
        offs_col + 1, 1, 1)).transpose((1, 0, 2))
    # print imgcanvas.shape
    # cv2.imshow("t", imgcanvas)
    # cv2.waitKey(0)
    imrange_rescale = cv2.resize(imgcanvas, (NNSIZE[0], NNSIZE[1]),
                                 interpolation=cv2.INTER_CUBIC)

    return (imrange_rescale)