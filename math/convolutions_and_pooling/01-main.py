#!/usr/bin/env python3

import numpy as np
convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid

np.random.seed(0)
m = np.random.randint(1000, 2000)
h, w = np.random.randint(100, 200, 2).tolist()
fh, fw = np.random.randint(3, 10, 2).tolist()

images = np.random.randint(0, 256, (m, h, w))
kernel = np.random.randint(0, 10, (fh, fw))
conv_ims = convolve_grayscale_valid(images, kernel)
print(conv_ims)
print(conv_ims.shape)
