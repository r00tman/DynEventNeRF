import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ColorSpace:
    @staticmethod
    def to_linear(input):
        # from [0,1] non-linear to [0,1] linear
        return NotImplementedError()

    @staticmethod
    def from_linear(input):
        # from [0,1] linear to [0,1] non-linear
        return NotImplementedError()


class Linear(ColorSpace):
    @staticmethod
    def to_linear(input):
        return input

    @staticmethod
    def from_linear(input):
        return input


class SRGB(ColorSpace):
    @staticmethod
    def to_linear(input):
        linmask = input <= 0.04045
        linval = input/12.92
        expval = ((input+0.055)/1.055)**2.4
        res = linval*linmask+(~linmask)*expval
        return res

    @staticmethod
    def from_linear(input):
        linmask = input <= 0.0031308
        linval = input*12.92
        expval = (input**(1/2.4))*1.055-0.055
        res = linval*linmask+(~linmask)*expval
        return res


class Gamma22(ColorSpace):
    @staticmethod
    def to_linear(input):
        # from [0,1] non-linear to [0,1] linear
        res = input ** 2.2
        return res

    @staticmethod
    def from_linear(input):
        # from [0,1] linear to [0,1] non-linear
        res = input ** (1/2.2)
        return res


class EventLogSpace(ColorSpace):
    @staticmethod
    def to_linear(input, eps):
        # from [0,1] non-linear to [0,1] linear
        res = torch.exp(input) - eps
        return res

    @staticmethod
    def from_linear(input, eps):
        # from [0,1] linear to [0,1] non-linear

        res = torch.log(input + eps)
        return res
