"""Module containing pre-processing algorithms to remove silence and noise from
utterances.

References:

[1] L. R. Rabiner. "Algorithm for determining the endpoints of isolated utterances."
in Journal of the Acoustical Society of America, volume 56, page S31, 1974.
"""


from basic import Wave
import math
import numpy as np



def frame_energy(wave, sample_len, index):
    """Calculates the energy (dB) of a frame. The function receives a Wave object,
    the frame length in samples and the frame index.
    """
    start = index * sample_len
    end = (index + 1) * sample_len
    if end > wave.length():
        end = wave.length()
    samples = wave.data[start : end]

    energy = 0
    for sample in samples:
        energy = energy + np.absolute(sample)  #'magnitude' instead of 'square' explained in [1]

    return energy

def energy(wave, frame_len=0.01):
    """Returns the energy of the signal (dB), given a Wave object and the frame length
    in seconds (default 0.02).
    """
    sample_len = math.ceil(wave.rate * frame_len)
    num_frames = math.ceil(wave.length() / sample_len)

    energy = np.zeros(num_frames)
    for i in range(num_frames):
        energy[i] = frame_energy(wave, sample_len, i)

    return energy

def signal(array):
    """Auxiliar function. Similar to 'numpy.sign', but with 'x == 0' returning '1'.
    """
    ret = [-1 if a < 0 else 1 for a in array]
    return np.array(ret, dtype=np.int64)

def frame_zcr(wave, sample_len, index):
    """Calculates the zero crossing rate (zcr) in a frame.
    """
    start = index * sample_len
    end = (index + 1) * sample_len
    if end > wave.length():
        end = wave.length()
    samples_signals = signal(wave.data[start : end])
    length = len(samples_signals)

    crossings = 0
    for n in range(1, length):
        crossings = crossings + abs(samples_signals[n] - samples_signals[n - 1])//2

    return crossings

def zcr(wave, frame_len=0.01):
    """Calculates the zero crossing rate (zcr) of the signal for each frame (by
    default 10 ms length).
    """
    sample_len = math.ceil(wave.rate * frame_len)
    num_frames = math.ceil(wave.length() / sample_len)

    zcr = np.zeros(num_frames)
    for i in range(num_frames):
        zcr[i] = frame_zcr(wave, sample_len, i)

    return zcr

def search_point(index, energy, itl, itu, incr=1):
    """Search for the beginning point (if incr == 1) or the ending point (if
    incr == -1).
    """
    m = index
    while True:
        while energy[m] < itl:
            m = m + incr

        i = m

        while (energy[i] >= itl) and (energy[i] < itu): #quebra em f00/phrase02, i = 10 (i == len)
            i = i + incr

        if energy[i] < itl:
            m = i + incr
        else:   #if "energy[i] >= itl", then "energy[i] >= itu"
            if i == m:
                return (i - incr)
            return i


def vad(wave, frame_len=0.01): #VAD energy + zcr
    wave_energy = energy(wave)
    wave_zcr = zcr(wave)

    initial_silence_len = math.ceil(0.1/frame_len)  #number of frames for the first 100 ms

    #zero crossing rate
    zcr_avg = np.average(wave_zcr[ : initial_silence_len])
    zcr_std = np.std(wave_zcr[ : initial_silence_len])
    fixed_izct = 25     #25 crossings for a frame of 10 ms
    izct = min(fixed_izct, zcr_avg + 2*zcr_std) #zero crossing threshold

    #energy
    energy_initial_silence = wave_energy[ : initial_silence_len]
    peak_energy = np.amax(energy_initial_silence)
    silence_energy = np.amin(energy_initial_silence)
    I1 = 0.03*(peak_energy - silence_energy) + silence_energy
    I2 = 4*silence_energy
    itl = min(I1, I2)   #lower energy threshold
    itu = 5*itl         #higher energy threshold

    #beginning point
    N1 = search_point(0, wave_energy[ : initial_silence_len], itl, itu)
    print('N1 =', N1)

    #ending point
    N2 = search_point(initial_silence_len - 1, wave_energy[ : initial_silence_len],
                      itl, itu, incr=-1)
    print('N2 =', N2)

    #TODO cortar o lixo


def draw(wave):
    """Draws the graphics of the signal,, energy, energy in dB and zero crossing rate.
    """
    wave_energy = energy(wave)
    wave_zcr = zcr(wave)

    plt.clf()
    plt.suptitle('SIGNAL')
    plt.grid(True)
    plt.plot(wave.data, 'b')
    plt.savefig('corpus_%s/%s/%s/%s.png' % (command, subcorpus, speaker, utterance))
    plt.clf()
    plt.suptitle('ENERGY')
    plt.grid(True)
    plt.plot(wave_energy, 'r')
    plt.savefig('corpus_%s/%s/%s/%s-energy.png' % (command, subcorpus, speaker, utterance))
    plt.clf()
    plt.suptitle('ENERGY (dB)')
    plt.grid(True)
    plt.plot(10*np.log10(wave_energy), 'm')
    plt.savefig('corpus_%s/%s/%s/%s-energy-dB.png' % (command, subcorpus, speaker, utterance))
    plt.clf()
    plt.suptitle('ZERO CROSSING RATE')
    plt.grid(True)
    plt.plot(wave_zcr, 'g')
    plt.savefig('corpus_%s/%s/%s/%s-zcr.png' % (command, subcorpus, speaker, utterance))


if __name__ == '__main__':
    import os.path
    import shutil
    import sys
    import matplotlib.pyplot as plt
    from basic import read_speakers, read_utterances_from_speaker, read_utterance


    if len(sys.argv) <= 1:
        print('write something you fool!')
        sys.exit()
    command = sys.argv[1]   # 'vad', 'draw'

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

                if command == 'vad':
                    vad(wave)
                elif command == 'draw':
                    draw(wave)

    print('finished')
