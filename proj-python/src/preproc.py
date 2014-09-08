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
        energy = energy + numpy.absolute(wave.data[n])

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

def vad(wave, threshold): #VAD energy + zero crossing rate
    """Remove signal frames when the energy is lower than a certain threshold.
    """
    pass


if __name__ == '__main__':
    import os.path
    import shutil
    import sys
    import matplotlib.pyplot as plt
    from basic import read_speakers, read_utterances_from_speaker, read_utterance


    if len(sys.argv) <= 1:
        print('write something you fool!')
        sys.exit()
    command = sys.argv[1]   # 'draw', 'vad'

    if os.path.exists('corpus_%s' % command):
        shutil.rmtree('corpus_%s' % command)
    os.mkdir('corpus_%s' % command)

    corpus = ['enroll_1', 'enroll_2', 'imposter']
    for subcorpus in corpus:
        os.mkdir('corpus_%s/%s' % (command, subcorpus))

        subcorpus_path = 'corpus/%s/' % subcorpus
        print(subcorpus_path)
        (females, males) = read_speakers(subcorpus_path)
        speakers = females + males

        for speaker in speakers:
            os.mkdir('corpus_%s/%s/%s' % (command, subcorpus, speaker))

            print(speaker)
            utterances = read_utterances_from_speaker(subcorpus_path, speaker)
            for utterance in utterances:
                print(utterance)
                wave = read_utterance(subcorpus_path, speaker, utterance)
                wave_energy = energy(wave)
                #TODO calcular zero crossing rate

                if command == 'draw':
                    plt.clf()
                    plt.suptitle('SIGNAL')
                    plt.grid(True)
                    plt.plot(wave.data, 'r')
                    plt.savefig('corpus_%s/%s/%s/%s-A.png' % (command, subcorpus, speaker, utterance))
                    plt.clf()
                    plt.suptitle('ENERGY (dB)')
                    plt.grid(True)
                    plt.plot(wave_energy, 'b')
                    plt.savefig('corpus_%s/%s/%s/%s-B.png' % (command, subcorpus, speaker, utterance))

    print('finished')
