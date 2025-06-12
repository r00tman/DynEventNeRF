#!/usr/bin/env python3
import numpy as np
from scipy import optimize as opt


def distort_norm(x, y, distortion):
    k1, k2, p1, p2, k3 = distortion
    x2 = x*x
    y2 = y*y
    r2 = x2 + y2
    _2xy = 2*x*y
    kr = 1 + ((k3*r2 + k2)*r2 + k1)*r2
    xd = x*kr + p1*_2xy + p2*(r2 + 2*x2)
    yd = y*kr + p1*(r2 + 2*y2) + p2*_2xy

    return xd, yd


def undistort_norm(xd, yd, distortion):
    def loss(p):
        x, y = p
        xd1, yd1 = distort_norm(x, y, distortion)
        return [xd1-xd, yd1-yd]

    x, y = opt.fsolve(loss, np.array([xd, yd]))
    return x, y


def distort_abs(x, y, camMatrix, distortion):
    f = camMatrix[0, 0]
    px, py = camMatrix[0, 2], camMatrix[1, 2]

    x = (x - px)/f
    y = (y - py)/f
    x, y = distort_norm(x, y, distortion)
    x = x*f+px
    y = y*f+py
    return x, y


def undistort_abs(x, y, camMatrix, distortion):
    f = camMatrix[0, 0]
    px, py = camMatrix[0, 2], camMatrix[1, 2]

    x = (x - px)/f
    y = (y - py)/f
    x, y = undistort_norm(x, y, distortion)
    x = x*f+px
    y = y*f+py
    return x, y


def mydistort(img):
    out = img[:]*0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            newx, newy = distort_abs(x+0.5, y+0.5, camMatrix, distortion)
            if 0 <= newx < img.shape[1] and 0 <= newy < img.shape[0]:
                out[int(newy), int(newx)] = img[y, x]
    return out


def myundistort(img):
    out = img[:]*0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            newx, newy = distort_abs(x+0.5, y+0.5, camMatrix, distortion)
            if 0 <= newx < img.shape[1] and 0 <= newy < img.shape[0]:
                out[y,x] = img[int(newy), int(newx)]
    return out


def mydistort_precise(img):
    out = img[:]*0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            newx, newy = undistort_abs(x+0.5, y+0.5, camMatrix, distortion)
            if 0 <= newx < img.shape[1] and 0 <= newy < img.shape[0]:
                out[y,x] = img[int(newy), int(newx)]
    return out


if __name__ == "__main__":
    import cv2
    frame = cv2.imread('r_00_0000.png')

    distortion = np.array([-0.332126588, -0.375395536, -0.001458110, 0.003408560, 1.431788445])

    # k1, k2 = -0.332126588, -0.375395536
    # p1, p2 = -0.001458110, 0.003408560
    # k3 = 1.431788445
    # distortion = np.array([k1, k2, p1, p2, k3])

    # messed up
    # k1, k2, k3 =  -0.430869192, 0.271721691, -0.109528683
    # p1, p2 = -0.001458110, 0.003408560

    camMatrix = np.array([
           [0.767040312,  0.000000000,  0.478860527],
           [0.000000000,  0.767040312,  0.378568679],
           [0.000000000,  0.000000000,  1.000000000],
    ])

    camMatrix[:2] *= frame.shape[1]


    undist = cv2.undistort(frame, camMatrix, distortion)
    myundist = myundistort(frame)
    myundistdist = mydistort_precise(myundist)
    cv2.imshow('dist', frame)
    cv2.imshow('myundist', myundist)
    # cv2.imshow('diff', (127+(undist/4-frame/4)).astype(np.uint8))
    cv2.imshow('mydist', myundistdist)
    cv2.imshow('diff', (127+(frame/4-myundistdist/4)).astype(np.uint8))
    # cv2.imshow('myundist', myundistort(frame))
    # cv2.imshow('diff', (127+(undist/4-myundistort(frame)/4)).astype(np.uint8))
    cv2.waitKey(0)

    # opt.fsolve(lambda x: disto

