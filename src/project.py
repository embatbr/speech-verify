#!/usr/bin/python3.4


"""Project's main module. In this file is where all codes are used to perform
the task of speaker verification.
"""


import numpy as np
import scipy.io.wavfile as wavfile
import os

import features


CORPUS_PATH = '../corpus/'


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


def read_speakers(dirname):
    dirs = os.listdir('%s/%s/' % (CORPUS_PATH, dirname))

    females = [d for d in dirs if d[0] == 'f']
    females.sort()
    males = [d for d in dirs if d[0] == 'm']
    males.sort()

    return (females, males)

def read_utterances(speaker):
    speaker_list = os.listdir('%s/enroll_1/%s/' % (CORPUS_PATH, speaker))[0]
    utterances = os.listdir('%s/enroll_1/%s/%s/' % (CORPUS_PATH, speaker, speaker_list))
    utterances = [utterance for utterance in utterances if utterance.endswith('.wav')]
    utterances.sort()

    return utterances


def enroll_1():
    (females, males) = read_speakers('enroll_1')
    # lists of list
    female_utterances = [read_utterances(female) for female in females]
    male_utterances = [read_utterances(male) for male in males]

    # GMM female
    for utterances in female_utterances:
        # TODO fazer um "for utterance in utterances: do GMM"
        print(utterances)
        print()

    # GMM male
    for utterances in male_utterances:
        pass


def enroll_2():
    read_speakers('enroll_2')


def imposter():
    read_speakers('imposter')


if __name__ == '__main__':
    #enroll_1()
    #enroll_2()
    #imposter()

    wave = Wave('test.wav')
    print('signal with', len(wave.signal), 'samples')
    print()

    framedMFCCs = features.mfcc(wave.signal, numcep=19, highfreq=wave.sample_rate/2)
    framedMFCCsDelta = features.appendDeltasAllFrames(framedMFCCs)
    framedMFCCsDeltaDelta = features.appendDeltasAllFrames(framedMFCCs, order=2)

    print('framedMFCCs:', framedMFCCs.shape)
    print(framedMFCCs)
    print('framedMFCCs + delta:', framedMFCCsDelta.shape)
    print('framedMFCCs + delta + delta-delta:', framedMFCCsDeltaDelta.shape)