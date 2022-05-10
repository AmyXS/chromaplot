# Audioplot

Python tools that plot spectrograms and chromagrams for a given audio file

Use python 3.9.12

## Setup 
```bash
py -m pip install librosa scipy matplotlib ipython numpy # Windows

pip3 install librosa scipy matplotlib ipython
pip3 install --no-binary numpy # this is required for some macOS problems
```

## Running
Use the VSCode run button on the top right. Make sure the right python version is selected. If not, go to CMD Shift P: >python: Select Interpreter

```
python3 chromaprint.py
```


## References

Super simple guide on plotting spectrogram
https://pythontic.com/visualization/signals/spectrogram

https://www.audiolabs-erlangen.de/resources/MIR/FMP/B/B_PythonVisualization.html

https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S1_SpecLogFreq-Chromagram.html