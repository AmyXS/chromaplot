# %matplotlib inline
import scipy.io, scipy, matplotlib.pyplot as plt, IPython.display as ipd
import librosa, librosa.display

# https://musicinformationretrieval.com/chroma.html

x, sr = librosa.load('audio/simple_piano.wav')
ipd.Audio(x, rate=sr)

# To compute a constant-Q spectrogram, will use librosa.cqt
fmin = librosa.midi_to_hz(36)
hop_length = 512
C = librosa.cqt(x, sr=sr, fmin=fmin, n_bins=72, hop_length=hop_length)

#Display:
logC = librosa.amplitude_to_db(numpy.abs(C))
plt.figure(figsize=(15, 5))
librosa.display.specshow(logC, sr=sr, x_axis='time', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
