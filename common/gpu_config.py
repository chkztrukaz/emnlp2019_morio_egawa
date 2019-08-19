# coding:utf-8

from chainer import cuda
import numpy as np
from . import config as cfg


'''
-----------------------------------------------------------------------------------------------------
GPU configuration
-----------------------------------------------------------------------------------------------------
'''
xp = np  # xp
GPU = 3  # GPU id
disable_gpu = False

try:
    if disable_gpu is not True:
        import cupy
        xp = cuda.cupy
except Exception as e:
    GPU = -1
    disable_gpu = True

if disable_gpu is not True:
    print('{}Cupy enabled.{}'.format(cfg.COL_INFO, cfg.COL_END))
    xp = cuda.cupy
else:
    print('{}Cupy is not available; did you try \'pip install cupy\'?{}'.format(cfg.COL_FAIL, cfg.COL_END))
    print('{}Cupy disabled.{}'.format(cfg.COL_INFO, cfg.COL_END))
    xp = np


def set_gpu(gpu_num: int):
    global GPU
    global xp
    global disable_gpu
    if gpu_num < 0:
        xp = np
        GPU = -1
        disable_gpu = True
    else:
        GPU = gpu_num

