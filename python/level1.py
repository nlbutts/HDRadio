# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 05:56:24 2018

@author: Nick Butts
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

def plotFFT(x, fs, nfft=8192):
    X = np.fft.fft(x)    
    W = np.linspace(-fs/2, fs/2, len(X))
    X = np.fft.fftshift(X)
    plt.clf()
    plt.plot(W, X)
    plt.grid()
    
def plotSpec(x, fs):
    f, t, Sxx = signal.spectrogram(x, fs)
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    plt.clf()
    plt.pcolormesh(t, f, np.log(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    plt.show()

def plotFilter(b, fs):
    w, h = signal.freqz(b)
    f = (w/2*np.pi) * fs
    fig = plt.figure()
    plt.title('Digital filter frequency response')
    ax1 = fig.add_subplot(111)

    plt.plot(f, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(f, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.show()

def filterRF(x):
    data = np.load('bpf.npz')
    ba = data['ba']
    b = ba[0,:]
    b = b + 1j*b
    return signal.lfilter(b, 1, x)

def autocorrelate(x, real=False, start=0, end=2160*4):
    x = x[start:end]
    if (real):
        x = np.real(x)
    y = signal.correlate(x, x[start:end], mode='same')
    return y[y.size//2:]

fs = 744187.5 * 2
ofdm_samples = 2160

#x = np.fromfile("../data/sample", dtype='uint8')
x = np.fromfile("../data/98.7", dtype='uint8')
x = x[0:1000000]
x = x.astype('float')
# neg_numbers = x > 127
# neg_index  =np.nonzero(neg_numbers)
# x[neg_index] = x[neg_index] - 256
#x /= 128
x -= 127
x *= 0.008

y = x[0::2] - 1j*x[1::2]

z = filterRF(y)
plotSpec(y, fs)
plotSpec(z, fs)
zd = z[0::2]
plotSpec(zd, fs/2)
end = 2048*4

f,ax=plt.subplots(2,1, sharex=True)
zc = autocorrelate(zd, start=2160)
ones = np.ones(ofdm_samples)
zcf = np.convolve(zc, ones, mode='same')/len(ones)
ax[0].plot(zc)
ax[0].grid()
ax[1].plot(zcf)
ax[1].grid()

