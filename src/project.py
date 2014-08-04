#!/usr/bin/python3.4


"""Project's main module. In this file is where all codes are used to perform
the task of speaker verification.
"""


import numpy as np
import scipy.io.wavfile as wavfile
import os

import features


ENROLL_1 = '../corpus/enroll_1'
ENROLL_2 = '../corpus/enroll_2'
IMPOSTER = '../corpus/imposter'


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


def read_speakers(basepath):
    dirs = os.listdir(basepath)

    females = [d for d in dirs if d[0] == 'f']
    females.sort()
    males = [d for d in dirs if d[0] == 'm']
    males.sort()

    return (females, males)

def read_utterances(basepath, speaker):
    speaker_list = os.listdir('%s/%s/' % (basepath, speaker))[0]
    utterances = os.listdir('%s/%s/%s/' % (basepath, speaker, speaker_list))
    #utterances = [utterance for utterance in utterances if utterance.endswith('.wav')]
    utterances = [utterance for utterance in utterances]
    utterances.sort()

    return utterances

def base_features(basepath):
    (females, males) = read_speakers(basepath)
    # list of list
    female_utterances_list = [read_utterances(basepath, female) for female in females]
    male_utterances_list = [read_utterances(basepath, male) for male in males]

def enroll_1():
    base_features(ENROLL_1)

# TODO fazer o mesmo para enroll_2 e imposter


if __name__ == '__main__':
    #enroll_1()
    enroll_2()
    imposter()

    wave = Wave('test.wav')
    framedMFCCs = features.mfcc(wave.signal, samplerate=wave.sample_rate,
                                numcep=19, highfreq=wave.sample_rate/2)
    framedMFCCsDelta = features.appendDeltasAllFrames(framedMFCCs)
    framedMFCCsDeltaDelta = features.appendDeltasAllFrames(framedMFCCs, order=2)

    print('framedMFCCs:', framedMFCCs.shape)
    print('framedMFCCs + delta:', framedMFCCsDelta.shape)
    print('framedMFCCs + delta + delta-delta:', framedMFCCsDeltaDelta.shape)