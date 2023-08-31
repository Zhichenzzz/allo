# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-builtin, unused-argument

import numpy as np


def grid(*args, name=None):
    return np.ndindex(*args)


def reduction(*args, name=None):
    return np.ndindex(*args)


def matmul(lhs, rhs, name=None):
    return np.matmul(lhs, rhs)


def bmm(lhs, rhs, name=None):
    return np.einsum("ijk,ikn->ijn", lhs, rhs)


def add(lhs, rhs, name=None):
    return lhs + rhs


def sub(lhs, rhs, name=None):
    return lhs - rhs


def div(lhs, rhs, name=None):
    return lhs / rhs


def copy(x, name=None):
    return np.copy(x)


def transpose(x, axes, name=None):
    if len(axes) == 2 and len(axes) < len(x.shape):
        new_axes = []
        for i in range(len(x.shape)):
            new_axes.append(i)
        new_axes[axes[0]] = axes[1] if axes[1] >= 0 else len(x.shape) + axes[1]
        new_axes[axes[1]] = axes[0] if axes[0] >= 0 else len(x.shape) + axes[0]
        axes = new_axes
    return np.transpose(x, axes)


def exp(x, name=None):
    return np.exp(x)


def log(x, name=None):
    return np.log(x)


def log2(x, name=None):
    return np.log2(x)


def log10(x, name=None):
    return np.log10(x)


def abs(x, name=None):
    return np.abs(x)


def softmax(x, name=None):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sqrt(x, name=None):
    return np.sqrt(x)


def sin(x, name=None):
    return np.sin(x)


def cos(x, name=None):
    return np.cos(x)


def tan(x, name=None):
    return np.tan(x)


def tanh(x, name=None):
    return np.tanh(x)


def power(x, y, name=None):
    return np.power(x, y)


def relu(x, name=None):
    return np.maximum(x, 0)


def linear(X, A, B, name=None):
    # TODO: Handle bias=None
    return matmul(X, A.T) + B


def view(x, shape, name=None):
    return np.reshape(x, shape)
