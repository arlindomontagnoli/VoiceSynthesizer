import numpy as np
import matplotlib.pyplot as plt
import time 
import scipy.signal as sig
import scipy.fft as fft
import sounddevice as sd

def Rosenberg(N1, N2,  f0, fs):
            T = 1 / f0;    # period in seconds
            pulseLen = int(np.floor(T * fs));    # length of one period of pulse
            # select N1 and N2 for duty cycle
            N2 = int(np.floor(pulseLen * N2))
            N1 = int(np.floor(N1 * N2))
            #hg = new double[(int)pulselength]
            hg = np.zeros(pulseLen)
            # calculate pulse samples
            for n in range(N1):
                hg[n] = 0.5 * (1 - np.cos(np.pi * (n) / N1))

            for  n in range(N1,N2):
                hg[n] = np.cos(np.pi * (n - N1) / (N2 - N1) / 2)
            return(hg)
#-----------------------------------------------------------------
# Voice paramters
#-----------------------------------------------------------------
fo = 120      #fundamental frequency(Hz)
synthTime = 5 #synthesis time (s)
Fs = 44100    #Sampling frequency (S/s)
#-----------------------------------------------------------------
# 2nd Order perturbation paramters
#-----------------------------------------------------------------
gain = 0.05
Wn = 4.0
epsilon = 0.4 
#-----------------------------------------------------------------
# Random pertubation paramter
#-----------------------------------------------------------------
randGain = 0.1
#-----------------------------------------------------------------
# Glottal model parameters
#-----------------------------------------------------------------
glottalOpen = 0.7   #Glottal openning time (% of glottal oppen)
dutyCycle = 0.8 #duty Cycle (% of period)  
n = 2048        # points
#-----------------------------------------------------------------
# Vocal Tract model paramters
#-----------------------------------------------------------------
F1 = 750        #1st formant
B1 = 75         #1st band
F2 = 1280        #2nd formant
B2 = 95         #2nd band
F3 = 2480        #3th formant
B3 = 150         #3th band
F4 = 3570        #4th formant
B4 = 400         #4th band
#-----------------------------------------------------------------
# Lip model paramter
#-----------------------------------------------------------------
mu = 0.98
#-----------------------------------------------------------------
#-----------------------------------------------------------------
# Algorithm
#-----------------------------------------------------------------
T = 1/Fs
npo = int(Fs/fo) #points per period
jitterLen = int(Fs*synthTime/npo)

np.random.seed(seed=int(time.time())) 
jitter = np.zeros(jitterLen)
y = np.zeros(jitterLen)
t = np.zeros(jitterLen)
B = 1/np.sqrt(1-epsilon)
sum = 0
jitter[0] = npo
for i in range(1,jitterLen):
    rand = (np.random.rand()-0.5)*randGain*10
    t[i] = synthTime*i/jitterLen
    y[i] = gain*(Wn/B)*(np.exp(-epsilon*Wn*t[i]))*np.sin(Wn*B*t[i])
    jitter[i] = jitter[i-1] + rand + y[i] 
    sum = sum + jitter[i]
mean = sum/(jitterLen-1)
jitterRel = jitter-mean

plt.plot(t,jitterRel)
plt.grid()


hg = Rosenberg(glottalOpen, dutyCycle, fo, Fs)

'''
plt.plot(hg)
plt.grid()
plt.show()
'''
hg_ts = np.zeros(n)     #hg pulse tain sample
excit_ts = np.zeros(n)  #excitation pulse train sample
hgLen = np.size(hg)
for i in range(n):
    hg_ts[i] = hg[i%hgLen]
    if i%hgLen==0:
         excit_ts[i] = 1
'''
plt.plot(excit_ts)
plt.grid()
plt.show()
plt.plot(hg_ts)
plt.grid()
plt.show()
'''

#-----------------------------------------------------------------
# Lip radiation model
#-----------------------------------------------------------------
lip_ts = np.zeros(n)
lip_ts[0] = 1
lip_ts[1] = -mu

a = 1.0
b = [1.0, -mu]
x = np.zeros(n)
x[0] = 1.0
hl = sig.lfilter(b,a,x)

'''
yf = fft.fft(lip_ts)
xf = fft.fftfreq(n, 4*T)[:n//2]
plt.yscale("log")
plt.plot(xf,2.0/n * np.abs(yf[0:n//2]))
plt.grid()
plt.show()
'''
#-----------------------------------------------------------------
# Vocal tract model
#-----------------------------------------------------------------
a1 = -2 * np.exp(-np.pi * B1 * T) * np.cos(2 * np.pi * F1 * T)
a2 = np.exp(-2 * np.pi * B1 * T)
a = [1.0, a1, a2]
b = [1.0]
y1 = sig.lfilter(b,a,x)
#-----------------------------------------------------------------
a1 = -2 * np.exp(-np.pi * B2 * T) * np.cos(2 * np.pi * F2 * T)
a2 = np.exp(-2 * np.pi * B2 * T)
a = [1.0, a1, a2]
b = [1.0]
x = np.zeros(n)
x[0] = 1.0
y2 = sig.lfilter(b,a,x)
#-----------------------------------------------------------------
a1 = -2 * np.exp(-np.pi * B3 * T) * np.cos(2 * np.pi * F3 * T)
a2 = np.exp(-2 * np.pi * B3 * T)
a = [1.0, a1, a2]
b = [1.0]
x = np.zeros(n)
x[0] = 1.0
y3 = sig.lfilter(b,a,x)
#-----------------------------------------------------------------
a1 = -2 * np.exp(-np.pi * B4 * T) * np.cos(2 * np.pi * F4 * T)
a2 = np.exp(-2 * np.pi * B4 * T)
a = [1.0, a1, a2]
b = [1.0]
x = np.zeros(n)
x[0] = 1.0
y4 = sig.lfilter(b,a,x)
#-----------------------------------------------------------------
ht1 = np.convolve(y1,y2)
ht2 = np.convolve(ht1,y3)
ht =np.convolve(ht2,y4)

yf = fft.fft(ht)
xf = fft.fftfreq(n, 4*T)[:n//2]
'''
plt.yscale("log")
plt.plot(xf,2.0/n * np.abs(yf[0:n//2]))
plt.grid()
plt.show()
'''
#-----------------------------------------------------------------
# Voice Synthesis
#-----------------------------------------------------------------
simLen = jitterLen * npo
excitation = np.zeros(simLen)
j = 0.0
c = 0 
while c < jitterLen:        #excitation with perturbation 
    j = j + jitter[c]
    p = int(np.round(j))
    if p < simLen:
        excitation[p] = 1
    c= c + 1
#------------------------------------------
# S(z) = E(z) . G(z) . V(z) . L(z)
# s(n) = e(n) * hg(n) * hv(n) * hl(n)
#------------------------------------------
Glottal = np.convolve(excitation, hg)
Vocaltract = np.convolve(Glottal, ht)
s = np.convolve(Vocaltract, hl)
#------------------------------------------
s = s - np.mean(s)              # remove dc level
voice = s/np.max(np.abs(s))     # normalize
volume = 0.2
sd.play(volume*voice, Fs)
sd.wait()
plt.show()