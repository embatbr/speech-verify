"""This module contains basic codes for the project, such as a Wave object to
save the rate and data from a .wav files and functions to read the speakers and
the utterances.
"""


import scipy.io.wavfile as wavfile
import numpy as np
import os


class Wave(object):
    """Class to describe a basic .wav file. The Wave object contains the sampling
    rate and data from the .wav file.
    """
    def __init__(self, filename=None):
        if filename is not None:
            wavf = wavfile.read(filename)
            self.rate = wavf[0]
            self.data = wavf[1].astype(np.int64, copy=False) # by default, numpy creates a array of int16

    def __str__(self):
        ret = 'rate: %d\nsample_length: %d\ndata: %s' % (self.rate, self.length(),
              self.data)
        return ret

    def save(self, filename):
        """Saves a Wave object into a .wav file.
        """
        wavfile.write(filename, self.rate, self.data)

    def length(self):
        """Size of the data array.
        """
        return len(self.data)

    def clone(self, start, end):
        cloned_wave = Wave()
        cloned_wave.rate = self.rate
        cloned_wave.data = self.data[start : end].astype(np.int16, copy=False)
        return cloned_wave


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


# Test
if __name__ == '__main__':
    import os.path


    if not os.path.exists('tests'):
        os.mkdir('tests')
    basic = open('tests/basic.out', 'w')
    corpus = ['corpus/enroll_1/', 'corpus/enroll_2/', 'corpus/imposter/']

    for subcorpus in corpus:
        print(subcorpus)
        print(subcorpus, file=basic)
        (females, males) = read_speakers(subcorpus)
        speakers = females + males

        for speaker in speakers:
            print(speaker)
            print(speaker, file=basic)
            utterances = read_utterances_from_speaker(subcorpus, speaker)
            for utterance in utterances:
                print(utterance, file=basic)
                wave = read_utterance(subcorpus, speaker, utterance)
                print(wave, file=basic)

        print(file=basic)
        print(file=basic)

    print('output to file "tests/basic.out"')
