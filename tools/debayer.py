#!/usr/bin/env python3
import numpy as np
import cv2

def debayer_image(a):
    if len(a.shape) == 2:
        a = a[..., None]

    if a.shape[-1] == 3:
        a = a[..., 0]

    assert len(a.shape) == 3 and a.shape[-1] == 1

    default = cv2.cvtColor(a, cv2.COLOR_BayerBG2RGB)
    vng = cv2.cvtColor(a, cv2.COLOR_BayerBG2RGB_VNG)

    # correct vng y=-2 shift
    result = np.copy(default)
    result[2:] = vng[:-2]

    return result
