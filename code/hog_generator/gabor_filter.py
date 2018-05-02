# Code from here is used from the examples and samples here:
# https://github.com/opencv/opencv/blob/master/samples/python/gabor_threads.py
# Edited to turn it into a class

import numpy as np
import cv2 as cv
from multiprocessing.pool import ThreadPool


class GaborFilter:
    def __init__(self, sigma=4.0, lambda_param=10.0, gamma=0.5, psi=0, threads=8):
        self.sigma = sigma
        self.param_lambda = lambda_param
        self.gama = gamma
        self.psi = psi
        self.threads = threads
        self.filters = self.build_filters()

    @property
    def sigma(self):
        return self.__sigma

    @sigma.setter
    def sigma(self, value):
        self.__sigma = value

    def build_filters(self):
        filters = []
        ksize = 31
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)

        return filters

    def process(self, img):
        accum = np.zeros_like(img)
        for kern in self.filters:
            fimg = cv.filter2D(img, cv.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum

    def process_threaded(self, img):
        accum = np.zeros_like(img)

        def f(kern):
            return cv.filter2D(img, cv.CV_8UC3, kern)

        pool = ThreadPool(processes=self.threads)
        for fimg in pool.map(f, self.filters):
            np.maximum(accum, fimg, accum)
        return accum
