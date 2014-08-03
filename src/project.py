"""Project's main module. In this file is where all codes are used to perform
the task of speaker verification.
"""


import numpy as np
import scipy.io.wavfile as wavfile

import features


class Wave(object):
    """Class to describe a basic .wav file
    """
    def __init__(self, filename):
        wavf = wavfile.read(filename)
        self.sample_rate = wavf[0]
        self.signal = wavf[1]

    def __str__(self):
        ret = 'sample_rate: %d\nsample_width: %d\nsignal: %s' % (self.sample_rate,
              self.get_sample_width(), self.signal)
        return ret

    def get_sample_width(self):
        return len(self.signal)


if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    import scipy.signal as signal

    if len(sys.argv) != 2:
        print('Which file?')
        sys.exit(-1)

    filename = sys.argv[1]
    wave = Wave('%s.in.wav' % filename)
    print('signal with', len(wave.signal), 'samples')
    print()

    framedMFCCs = features.mfcc(wave.signal, numcep=19, highfreq=wave.sample_rate/2)
    framedMFCCsDelta = features.appendDeltasAllFrames(framedMFCCs)
    framedMFCCsDeltaDelta = features.appendDeltasAllFrames(framedMFCCs, order=2)

    print('framedMFCCs:', framedMFCCs.shape)
    print(framedMFCCs)
    print()
    print('framedMFCCsDelta:', framedMFCCsDelta.shape)
    print(framedMFCCsDelta)
    print()
    print('framedMFCCsDeltaDelta:', framedMFCCsDeltaDelta.shape)
    print(framedMFCCsDeltaDelta)

    #plt.grid(True)
    #plt.xlabel('sample')
    #plt.ylabel('intensity')
    #plt.plot(wave.signal)
    #plt.show()
