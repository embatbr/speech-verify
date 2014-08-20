"""This module contains basic codes for the project, such as a Wave object to
save the rate and data from .wav files and functions to read the utterances.
"""


import scipy.io.wavfile as wavfile
import os


class Wave(object):
    """Class to describe a basic .wav file. The Wave object contains the sampling
    rate and data from the .wav file.
    """
    def __init__(self, filename):
        wavf = wavfile.read(filename)
        self.rate = wavf[0]
        self.data = wavf[1]

    def __str__(self):
        ret = 'rate: %d\nsample_width: %d\ndata: %s' % (self.rate, self.width(),
              self.data)
        return ret

    def save(self, filename):
        wavfile.write(filename, self.rate, self.data)

    def width(self):
        return len(self.data)


def read_speakers(subcorpus):
    """Given a subcorpus (enroll_1, enroll_2 or imposter) it returns a tuple of
    names for speakers. The first element is a list of names for female speakers
    and the second is a list of names for male speakers.
    """
    dirs = os.listdir(subcorpus)

    females = [d for d in dirs if d[0] == 'f']
    females.sort()
    males = [d for d in dirs if d[0] == 'm']
    males.sort()

    return (females, males)

def read_utterances_from_speaker(subcorpus, speaker):
    """Given a subcorpus (enroll_1, enroll_2 or imposter) and a speaker (f00, f01,
    m00...) it returns the 54 utterances names (.wav) for this speaker in this
    particular base. The utterances are sorted.
    """
    utterances = os.listdir('%s/%s/' % (subcorpus, speaker))
    utterances = [utterance for utterance in utterances if utterance.endswith('.wav')]
    utterances.sort()

    return utterances

def read_utterance(subcorpus, speaker, utterance):
    wave = Wave('%s/%s/%s' % (subcorpus, speaker, utterance))
    return wave


if __name__ == '__main__':
    basic = open('basic.out', 'w')
    corpus = ['corpus/enroll_1/', 'corpus/enroll_2/', 'corpus/imposter/']

    for subcorpus in corpus:
        print(subcorpus, file=basic)
        (females, males) = read_speakers(subcorpus)

        for female in females:
            print(female, file=basic)
            utterances = read_utterances_from_speaker(subcorpus, female)
            for utterance in utterances:
                print(utterance, file=basic)
                wave = read_utterance(subcorpus, female, utterance)
                print(wave, file=basic)

        for male in males:
            print(male, file=basic)
            utterances = read_utterances_from_speaker(subcorpus, male)
            for utterance in utterances:
                print(utterance, file=basic)
                wave = read_utterance(subcorpus, male, utterance)
                print(wave, file=basic)

        print(file=basic)
        print(file=basic)
