import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import find_peaks, lfilter
from scipy.fftpack import fft
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
#from tkinter import Tk
from tkinter.filedialog import askopenfilename#, askopenfilenames
#-------------------------------------------------------------
#load the signal
file = askopenfilename(initialfile = "*.wav", title = "Select a wave file",
                        filetypes = ((".wav file",(".wav")),("all files",".*")))
sample_rate, signal = wav.read(file)
#-------------------------------------------------------------
#-------------------------------------------------------------
#calculate the fundamental frequency of the signal
#-------------------------------------------------------------
def calculate_fundamental_frequency(signal, sampling_rate):
    # Normalize the signal
    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))    
    # Compute the autocorrelation of the signal
    autocorrelation = np.correlate(signal, signal, mode='full')
    autocorrelation = autocorrelation[len(autocorrelation)//2:]
    # Find the max peaks of the autocorrelation
    peaks, _ = find_peaks(autocorrelation, height=0)
    # Find the highest peak
    peak_index = peaks[np.argmax(autocorrelation[peaks])]
    # Calculate the fundamental frequency
    fundamental_frequency = sampling_rate / peak_index
    return fundamental_frequency
#-------------------------------------------------------------
#convert stereo to mono
if len(signal.shape) == 2:     
    signal = signal.mean(axis=1)
#normalize the signal
signal = signal/max(signal)
#sample of the signal
length = len(signal)
l2=length//2
sampleSize = 2048
#One sample of middle of the signal
sample = signal[l2:l2+sampleSize]
#calculate the fundamental frequency
fo = calculate_fundamental_frequency(sample, sample_rate)
print('Fundamental frequency: %0.1f '% fo)

h_len = int(sample_rate//fo)
h = signal[l2:l2+h_len]
partialCorr = np.correlate(signal, h, mode='full')

distance = int(sample_rate // fo)-20
#-------------------------------------------------------------
#manual adjust of threshold
threshold = 20
#-------------------------------------------------------------
peaks, _ = find_peaks(partialCorr, distance=distance,height=threshold)
#plot the peaks in the partial correlation
plt.figure() 
#create the time axis in seconds
t = np.arange(0, len(partialCorr)/sample_rate, 1/sample_rate)
plt.plot(t,partialCorr)
#plot the peaks
plt.plot(peaks/sample_rate, partialCorr[peaks], "x")
plt.xlabel('time(s)')
plt.ylabel('Amplitude')
plt.title('Partial Correlation and Peaks')
plt.grid()

#find distaces between peaks
distances = np.diff(peaks) 
distances = distances- np.mean(distances)
#plot the distances between peaks (jitter)
plt.figure()  
#create the time axis in seconds
total = t[-1]   
t = np.arange(0,total, total/len(distances))
plt.plot(t,distances)
plt.grid()
plt.ylim(-20,20)
plt.title('Perturbation (jitter)')
plt.xlabel('time(s)')   
plt.ylabel('Amplitude')
#plt.show()
#filter the distances
b, a = [1], [1, -0.95]
distances = lfilter(b, a, distances)
#plot the filtered distances
plt.figure()
plt.plot(t,distances/10)
plt.grid()
plt.ylim(-20,20)
plt.title('Filtered Perturbation (jitter)')
plt.xlabel('time(s)')
plt.ylabel('Amplitude')
plt.show()
#-------------------------------------------------------------
#-------------------------------------------------------------

    
