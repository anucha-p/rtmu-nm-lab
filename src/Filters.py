import numpy as np
import math

def getButterworth_lowpass_filter(shape, cutoff=0.25, order=2):
    m, n = shape
    d0 = cutoff
    h = np.zeros((m, n))
    X = np.linspace(-1,1,shape[0])
    Y = np.linspace(-1,1,shape[0])
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            d = math.sqrt((x ** 2) + (y ** 2))
            h[i, j] = 1 / (1 + (d / d0) ** (2 * order))
    return h

def getButterworth_highpass_filter(shape, cutoff=0.25, order=2):
    m, n = shape
    d0 = cutoff
    h = np.zeros((m, n))
    X = np.linspace(-1,1,shape[0])
    Y = np.linspace(-1,1,shape[0])
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            d = math.sqrt((x ** 2) + (y ** 2))
            h[i, j] = 1 / (1 + (d0 / d) ** (2 * order))
    return h

def getHanning_filter(shape, cutoff=0.25):
    m, n = shape
    d0 = cutoff
    h = np.zeros((m, n))
    X = np.linspace(-1,1,shape[0])
    Y = np.linspace(-1,1,shape[0])
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            d = math.sqrt((x ** 2) + (y ** 2))
            if 0<= d and d<=d0:
                h[i, j] = 0.5 + 0.5*math.cos(math.pi*d/d0)
            else:
                h[i, j] = 0
    return h

def getHamming_filter(shape, cutoff=0.25):
    m, n = shape
    d0 = cutoff
    h = np.zeros((m, n))
    X = np.linspace(-1,1,shape[0])
    Y = np.linspace(-1,1,shape[0])
    for i, x in enumerate(X):
        for j, y in enumerate(X):
            d = math.sqrt((x ** 2) + (y ** 2))
            if 0<= d and d<=d0:
                h[i, j] = 0.54 + 0.46*math.cos(math.pi*d/d0)
            else:
                h[i, j] = 0
    return h


def fourier_filter(image, filt):
    image_fft = np.fft.fft2(image)
    shift_fft = np.fft.fftshift(image_fft)
    filtered_image = np.multiply(filt, shift_fft)
    shift_ifft = np.fft.ifftshift(filtered_image)
    ifft = np.fft.ifft2(shift_ifft)
    filt_image = np.abs(ifft)
    return filt_image

def getGaussion_filter(shape=(3,3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h