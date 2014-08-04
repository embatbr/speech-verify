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
    """Given a basepath (enroll_1, enroll_2 or imposter) it returns a tuple of
    names for speakers. The first element is a list of names for female speakers
    and the second is a list of names for male speakers.
    """
    dirs = os.listdir(basepath)

    females = [d for d in dirs if d[0] == 'f']
    females.sort()
    males = [d for d in dirs if d[0] == 'm']
    males.sort()

    return (females, males)

def read_utterances(basepath, speaker):
    """Given a basepath (enroll_1, enroll_2 or imposter) and a speaker (f00, f01,
    m00...) it returns the 59 utterances names(.wav) for this speaker in this
    particular base. The utterances are sorted.
    """
    utterances = os.listdir('%s/%s/' % (basepath, speaker))
    utterances = [utterance for utterance in utterances if utterance.endswith('.wav')]
    utterances = [utterance for utterance in utterances]
    utterances.sort()

    return utterances

def read_utterances_files(basepath, utterances_list, gender):
    """Given a basepath (enroll_1, enroll_2 or imposter), a utterances_list (names
    of files) and the gender ('f' or 'm'), it reads the files containing the
    utterances
    """
    numSpeakers = len(utterances_list)
    utt_waves_list = list()

    for (speakerIndex, utterances) in zip(range(numSpeakers), utterances_list):
        utt_waves = list()
        for utterance in utterances:
            utt_wave = Wave('%s/%s%02d/%s' % (basepath, gender, speakerIndex,
                                              utterance))
            utt_waves.append(utt_wave)

        utt_waves_list.append(utt_waves)

    return utt_waves_list

def features_from_base(basepath, order=0):
    (females, males) = read_speakers(basepath)
    # list of list (sorted)
    female_utterances_list = [read_utterances(basepath, female) for female in females]
    male_utterances_list = [read_utterances(basepath, male) for male in males]

    # utterances as Wave objects
    female_utterances_list = read_utterances_files(basepath, female_utterances_list, 'f')
    male_utterances_list = read_utterances_files(basepath, male_utterances_list, 'm')

    for utterances in female_utterances_list:
        for utterance in utterances:
            uttMFCCs = features.mfcc(utterance.signal, samplerate=utterance.sample_rate,
                                     numcep=19, highfreq=utterance.sample_rate/2)
            if(order > 0):
                uttMFCCs = features.appendDeltasAllFrames(uttMFCCs, order)
            print(uttMFCCs.shape)
            print(uttMFCCs)

        print()

def enroll_1():
    features_from_base(ENROLL_1, order=2)

# TODO fazer o mesmo para enroll_2 e imposter


if __name__ == '__main__':
    enroll_1()

    wave = Wave('test.wav')
    framedMFCCs = features.mfcc(wave.signal, samplerate=wave.sample_rate,
                                numcep=19, highfreq=wave.sample_rate/2)
    framedMFCCsDelta = features.appendDeltasAllFrames(framedMFCCs)
    framedMFCCsDeltaDelta = features.appendDeltasAllFrames(framedMFCCs, order=2)

    print('framedMFCCs:', framedMFCCs.shape)
    print('framedMFCCs + delta:', framedMFCCsDelta.shape)
    print('framedMFCCs + delta + delta-delta:', framedMFCCsDeltaDelta.shape)