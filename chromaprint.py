
import matplotlib.pyplot as plot
import numpy as np
from scipy.io import wavfile

import librosa, librosa.display

# Read the wav file (mono)
samplingFrequency, signalData = wavfile.read('resources/simple_piano.wav')

def plot_amplitude():
    plot.subplot(211)
    plot.plot(signalData)
    plot.xlabel('Sample')
    plot.ylabel('Amplitude')

def plot_spectrogram():
    plot.subplot(212)
    plot.specgram(signalData,Fs=samplingFrequency)
    plot.xlabel('Time')
    plot.ylabel('Frequency')


# def compute_chromagram(Y_LF):
#     """Computes a chromagram

#     Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

#     Args:
#         Y_LF (np.ndarray): Log-frequency spectrogram

#     Returns:
#         C (np.ndarray): Chromagram
#     """
#     C = np.zeros((12, Y_LF.shape[1]))
#     p = np.arange(128)
#     for c in range(12):
#         mask = (p % 12) == c
#         C[c, :] = Y_LF[mask, :].sum(axis=0)
#     return C

# C = compute_chromagram(Y_LF)



# def plot_chromagram():
#     fig = plot.figure(figsize=(10, 3))
#     chroma_label = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
#     plot.imshow(10 * np.log10(eps + C), origin='lower', aspect='auto', cmap='gray_r', 
#             extent=[T_coef[0], T_coef[-1], 0, 12])
#     plot.clim([0, 60])
#     plot.xlabel('Time (seconds)')
#     plot.ylabel('Chroma')
#     cbar = plot.colorbar()
#     cbar.set_label('Magnitude (dB)')
#     plot.yticks(np.arange(12) + 0.5, chroma_label)
#     plot.tight_layout()

#     rect = matplotlib.patches.Rectangle((29.3, 0.0), 1.2, 12, linewidth=3, edgecolor='r', facecolor='none')
#     plot.gca().add_patch(rect)
#     plot.text(28.5, -1.2, r'$\mathrm{C3}$', color='r', fontsize='x-large');  


def plot_chromagram_librosa():
    plot.subplot(212)

    x, Fs = librosa.load('resources/simple_piano.wav', sr=samplingFrequency)
    N = 4096
    H = 1024
    eps = np.finfo(float).eps
    C = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)
    plot.figure(figsize=(8, 2))
    librosa.display.specshow(10 * np.log10(eps + C), x_axis='time', 
                            y_axis='chroma', sr=Fs, hop_length=H)
    plot.colorbar();

plot_amplitude()
plot_spectrogram()
plot_chromagram_librosa()

plot.show()