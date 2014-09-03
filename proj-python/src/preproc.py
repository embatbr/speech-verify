"""Module containing pre-processing algorithms to remove silence and noise from
utterances.
"""


from basic import Wave
import math
import numpy



def frame_energy(wave, sample_len, index):
    """Calculates the energy (dB) of a frame. The function receives a Wave object,
    the frame length in samples and the frame index.
    """
    start = index * sample_len
    end = (index + 1) * sample_len
    if end > wave.length():
        end = wave.length()
    samples = range(start, end)

    energy = 0
    for n in samples:
        energy = energy + numpy.power(wave.data[n], 2)

    energy = 10*numpy.log10(energy)
    return energy


def energy(wave, frame_len=0.01):
    """Returns the energy of the signal (dB), given a Wave object and the frame length
    in seconds (default 0.02).
    """
    sample_len = math.ceil(wave.rate * frame_len)
    num_frames = math.ceil(wave.length() / sample_len)

    energy = numpy.zeros(num_frames)
    for i in range(num_frames):
        energy[i] = frame_energy(wave, sample_len, i)

    return energy


if __name__ == '__main__':
    import os.path
    import sys
    import matplotlib.pyplot as plt
    from basic import read_speakers, read_utterances_from_speaker, read_utterance


    if not os.path.exists('tests'):
        os.mkdir('tests')
    if not os.path.exists('tests/preproc'):
        os.mkdir('tests/preproc')
    preproc = open('tests/preproc.out', 'w')

    if len(sys.argv) > 1:
        wave = read_utterance('corpus/%s/' % sys.argv[1], sys.argv[2], '%s_16k.wav' % sys.argv[3])
        print(wave.length())
        energy = energy(wave, frame_len=0.01)
        print(energy, file=preproc)
        plt.clf()
        plt.grid(True)
        plt.plot(wave.data, 'r')
        plt.savefig('tests/preproc/%s.%s.%s_16k.signal.png' % (sys.argv[1], sys.argv[2], sys.argv[3]))
        plt.clf()
        plt.grid(True)
        plt.plot(energy, 'b')
        plt.savefig('tests/preproc/%s.%s.%s_16k.energy_dB.png' % (sys.argv[1], sys.argv[2], sys.argv[3]))
    else:
        corpus = ['enroll_1', 'enroll_2', 'imposter']
        for subcorpus in corpus:
            subcorpus_path = 'corpus/%s/' % subcorpus
            print(subcorpus_path)
            print(subcorpus_path, file=preproc)
            (females, males) = read_speakers(subcorpus_path)
            speakers = females + males

            for speaker in speakers:
                print(speaker)
                print(speaker, file=preproc)
                utterances = read_utterances_from_speaker(subcorpus_path, speaker)
                for utterance in utterances:
                    print(utterance, file=preproc)
                    wave = read_utterance(subcorpus_path, speaker, utterance)
                    wave_energy = energy(wave)
                    print(wave_energy, file=preproc)

                    if len(sys.argv) > 1 and sys.argv[1] == 'draw':
                        plt.clf()
                        plt.grid(True)
                        plt.plot(wave.data, 'r')
                        plt.savefig('tests/preproc/%s-%s-%s-signal.png' % (subcorpus, speaker, utterance))
                        plt.clf()
                        plt.grid(True)
                        plt.plot(wave_energy, 'b')
                        plt.savefig('tests/preproc/%s-%s-%s-energy.png' % (subcorpus, speaker, utterance))

            print(file=preproc)
            print(file=preproc)

    print('output to file "tests/preproc.out"')
