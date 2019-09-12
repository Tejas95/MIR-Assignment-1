import numpy as np
from numpy import mean, sqrt, square, arange
from scipy.io import wavfile
import matplotlib.pyplot as plt 

# A.1
# LL 9/12/19: edited the block_audio function to make sure each block 
# starts 1 hop size after the last block instead of at the beginning of x
# Also divided the index by fs to get the timeInSec
def block_audio(x,blockSize,hopSize,fs):
    i=1
    xb=[]
    timeInSec=[]
    stop=0
    while stop<len(x):
        start=(i-1)*hopSize
        stop=start+blockSize
        i=i+1
        starttime=start/fs
        xb.append(x[start:stop])
        timeInSec.append(starttime)

    return np.array(xb), np.array(timeInSec)

# A.2 
'''
def comp_acf(inputVector, bIsNormalized):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]
'''

# A.3
'''
def get_f0_from_acf(r, fs):

'''

# A.4
def track_pitch_acf(x,blockSize,hopSize,fs):
	xb, timeInSec = block_audio(x,blockSize,hopSize,fs)
	r = comp_acf()
	f0 = get_f0_from_acf(r,fs)

	return f0, timeInSec

# B.1 (LL added 9/12/19)

# create 2 time vectors, 0 to 1 sec and 1 to 2 sec, each at a sample rate of 44.1kHz
fs=44100
timeA=np.linspace(start=0,stop=1,num=fs,endpoint=False)
timeB=np.linspace(start=1,stop=2,num=fs,endpoint=False)

# generate test signals at 441 Hz from 0 to 1 sec and 882 Hz from 1 to 2 sec
testsignalA=np.sin(2*np.pi*441*timeA)
testsignalB=np.sin(2*np.pi*882*timeB)

# append arrays to create a 2 sec test signal
time=np.append(timeA,timeB)
testsignal=np.append(testsignalA,testsignalB)   

# B.2 (LL added 9/12/19)
# TO DO: need to calculate the log without using the math library
import math

def convert_freq2midi(freqInHz):
  
    typ=type(freqInHz)
    
    #convert frequency to MIDI pitch for a scalar
    if typ==int or typ==float:
        pitchInMIDI=round(69+12*np.log(freqInHz/440,2))
   
    #do the same thing for a vector 
    #pitch function converts a single item, then np.vectorize applies it to an array
    else:    
        def pitch(x):
            return round(69+12*np.log(x/440,2))
    
        vectfunc=np.vectorize(pitch)
        pitchInMIDI=list(vectfunc(freqInHz))
    return pitchInMIDI
    

# B.3
def eval_pitchtrack(estimateInHz, groundtruthInHz):

	def hz2cents(freq_hz, base_frequency=10.0):
		freq_cent = np.zeros(freq_hz.shape[0])
	    freq_nonz_ind = np.flatnonzero(freq_hz)
	    normalized_frequency = np.abs(freq_hz[freq_nonz_ind])/base_frequency
	    freq_cent[freq_nonz_ind] = 1200*np.log2(normalized_frequency)
	    return freq_cent

	non_zero = (groundtruthInHz>0)

	error = groundtruthInHz[non_zero] - estimateInHz[non_zero]
	rms = sqrt(mean(square(error)))
	return rms

# B.4
def run_evaluation(path):

	def read_label(path):
		oup = []
		f = open(path, "r")
		for x in f:
  			oup.append(x.split('     ')[1])
  		return oup

	files = ['01-D_AMairena.f0.Corrected','24-M1_AMairena-Martinete.f0.Corrected','63-M2_AMairena.f0.Corrected']
	for file in files:
		fs, wav = wavfile.read('./trainData/'+file+'.wav')
		f0, timeInSec = track_pitch_acf(wav, blockSize, hopSize, fs)
		gtHz = read_label('./trainData/'+file+'.txt')
		rms = eval_pitchtrack(f0, gtHz)