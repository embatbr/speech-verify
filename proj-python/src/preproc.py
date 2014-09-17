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

def energy(wave, sample_len, num_frames):
    """Returns the energy of the signal (dB), given a Wave object and the frame length
    in seconds (default 0.02).
    """
    energy = np.zeros(num_frames)
    for i in range(num_frames):
        energy[i] = frame_energy(wave, sample_len, i)

    energy = energy.astype(np.int64, copy=False)
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

def zcr(wave, sample_len, num_frames):
    """Calculates the zero crossing rate (zcr) of the signal for each frame (by
    default 10 ms length).
    """
    zcr = np.zeros(num_frames)
    for i in range(num_frames):
        zcr[i] = frame_zcr(wave, sample_len, i)

    return zcr


def vad(wave, frame_len=0.01): #VAD energy + zcr
    sample_len = math.ceil(wave.rate * frame_len)
    num_frames = math.ceil(wave.length() / sample_len)
    wave_energy = energy(wave, sample_len, num_frames)
    wave_zcr = zcr(wave, sample_len, num_frames)

    initial_silence_len = math.ceil(0.1/frame_len)  #number of frames for the first 100 ms

    #zero crossing rate
    zcr_silence = wave_zcr[ : initial_silence_len]
    avg_zcr_silence = np.average(zcr_silence)
    std_zcr_silence = np.std(zcr_silence)
    fixed_izct = 25     #25 crossings for a frame of 10 ms
    izct = min(fixed_izct, avg_zcr_silence + 2*std_zcr_silence) #zero crossing threshold

    #energy
    energy_silence = wave_energy[ : initial_silence_len]
    IMX = np.amax(energy_silence)               #peak energy
    IMN = np.amin(energy_silence)               #silence energy
    I1 = 0.03*(IMX - IMN) + IMN
    I2 = 4*IMN
    ITL = min(I1, I2)   #lower energy threshold
    ITU = 5*ITL         #higher energy threshold

    #search for starting/ending point based on energy thresholds
    ITL_energy = [i for i in range(len(wave_energy)) if wave_energy[i] >= ITL]
    ITU_energy = [i for i in range(len(wave_energy)) if wave_energy[i] >= ITU]
    if (ITU_energy == []):  #if no frame energy exceeds ITU, there is no VAD
        return wave

    sample_len = math.ceil(wave.rate * frame_len)
    N1 = ITU_energy[0]
    if (N1 == ITL_energy[0]) and N1 > 0:
        N1 = N1 - 1
    N2 = ITU_energy[-1]
    if (N2 == ITL_energy[-1]) and N2 < (wave.length() - 1):
        n2 = N2 + 1
    print('U1 = 0 U2 =', len(wave_energy) - 1)
    print('N1 =', N1, 'N2 =', N2)

    return wave.clone(N1 * sample_len, N2 * sample_len)


def draw(wave, frame_len=0.01):
    """Draws the graphics of the signal,, energy, energy in dB and zero crossing rate.
    """
    sample_len = math.ceil(wave.rate * frame_len)
    num_frames = math.ceil(wave.length() / sample_len)
    wave_energy = energy(wave, sample_len, num_frames)
    wave_zcr = zcr(wave, sample_len, num_frames)

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
                    wave_vad = vad(wave)
                    wave_vad.save('corpus_%s/%s/%s/%s' % (command, subcorpus, speaker, utterance))
                elif command == 'draw':
                    draw(wave)

    print('finished')
