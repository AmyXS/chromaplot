
import matplotlib.pyplot as plot
import numpy as np
from scipy.io import wavfile

import librosa, librosa.display

# Read the wav file (mono)
audioPath ='resources/Feel_Like_Makin_Love/sing-1s.wav'
samplingFrequency, signalData = wavfile.read(audioPath)


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

N = 4096
H = 32
eps = np.finfo(float).eps
  
def plot_chromagram():
    plot.subplot(212)

    x, Fs = librosa.load(audioPath, sr=samplingFrequency)  
    C = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=None, hop_length=H)
    plot.figure(figsize=(8, 2))
    librosa.display.specshow(10 * np.log10(eps + C), x_axis='time', 
                            y_axis='chroma', sr=Fs, hop_length=H)
    plot.colorbar();
    
plot_amplitude()
plot_spectrogram()
plot_chromagram()

plot.show()