import numpy as np
from numpy import mean, sqrt, square, arange
from scipy.io import wavfile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt 

# A.1
def block_audio(x,blockSize,hopSize,fs):
    
    timeInSec=[]
    NumOfBlocks   = int(np.round(len(x)/hopSize))

    out = np.zeros((NumOfBlocks, blockSize))

    for n in range (NumOfBlocks):
        start     = n*hopSize
        stop      = min(len(x),start + blockSize)

        if stop - start < blockSize:

            vec = np.zeros(stop-start)

            vec[:] = np.array(x[start:stop])
            diff = blockSize - (stop-start)
            vec = np.append(vec, np.zeros(diff))
            
            out[n,:] = vec

        else:

            out[n,:] = x[start:stop]

        timeInSec.append(start/fs)

    return out,np.array(timeInSec);

# A.2 
# Amy, update out->inputVector
# Laney added normalization
def comp_acf(inputVector, bIsNormalized): 
    
    N = len(inputVector)
    ACF = np.zeros((N))

	#ACF

	for i in range (N):
		for j in range (N-i):
			ACF[i] = ACF[i] + inputVector[j] * inputVector[i+j]  
	
    if bIsNormalized=='TRUE':
        r=ACF/sum([x**2 for x in inputVector])
    else:
        r=ACF
       
    return r;

# A.3
def get_f0_from_acf(r, fs):
    
    peaks, _ = find_peaks(r, height=0)

	firstpeak = peaks[0]
	secondpeak = peaks[1]

	period = secondpeak - firstpeak
	
	f0 = np.float(fs/period)
	return f0;

# A.4
def track_pitch_acf(x,blockSize,hopSize,fs):
    xb, timeInSec = block_audio(x,blockSize,hopSize,fs)
	
	NumOfBlocks   = int(np.round(len(x)/hopSize))

	f0 = np.zeros((NumOfBlocks))

	for i in range (len(xb)):
		r = comp_acf(xb[i], None)
		f0[i] = get_f0_from_acf(r,fs)

	return f0, timeInSec

# B.1 
def sinusoidal_test():
    # create 2 time vectors, 0 to 1 sec and 1 to 2 sec, each at a sample rate of 44.1kHz
    fs=44100
    timeA=np.linspace(start=0,stop=1,num=fs,endpoint=False)
    timeB=np.linspace(start=1,stop=2,num=fs,endpoint=False)

    # generate test signals at 441 Hz from 0 to 1 sec and 882 Hz from 1 to 2 sec
    testsignalA=np.sin(2*np.pi*441*timeA)
    testsignalB=np.sin(2*np.pi*882*timeB)

    # append arrays to create a 2 sec test signal
    #time=np.append(timeA,timeB)
    testsignal=np.append(testsignalA,testsignalB)   

    f0, timeInSec = track_pitch_acf(testsignal,1024,512,fs)  
    
    err=np.zeros(len(f0))
    for i in range(len(f0)):
        if 0<=timeInSec[i]<1:
            err[i]=f0[i]-441
        elif timeInSec[i]>=1:
            err[i]=f0[i]-882
  
    #plot f0 
    
    plt.plot(timeInSec,f0)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Estimated Frequency (Hz)')
    plt.title('Estimated Frequency of Test Signal')
    plt.show
    
    plt.show(block=False)
    
    #plot error
    
    plt.plot(timeInSec,err)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Difference between Estimated and Actual Frequency (Hz)')
    plt.title('Error of Estimated Test Signal Frequency')
    plt.show
    
    plt.show(block=False)
    #output f0 to check the fundamental freqs assigned to each block
    return f0
   
sinusoidal_test()

# B.2 
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

    estimateInHz = hz2cents(estimateInHz)
    groundtruthInHz = hz2cents(groundtruthInHz)

    non_zero = (groundtruthInHz>0)
    error = groundtruthInHz[non_zero] - estimateInHz[non_zero]
    rms = sqrt(mean(square(error)))
    return rms

# B.4
def run_evaluation():

    blockSize = 1024
    hopSize = 512
    fs = 44100

    def read_label(path):
        oup = []
        f = open(path, "r")
        for x in f:
              oup.append(x.split('     ')[2])
        return oup

    files = ['01-D_AMairena','24-M1_AMairena-Martinete','63-M2_AMairena']
    for file in files:
        fs, wav = wavfile.read('./trainData/'+file+'.wav')
        f0, timeInSec = track_pitch_acf(wav, blockSize, hopSize, fs)
        
        gtHz = np.array(read_label('./trainData/'+file+'.f0.Corrected.txt')).astype(np.float)
        #f0 = np.load(file+'.f0.npy')
        rms = eval_pitchtrack(f0, gtHz) 
        print(rms)

run_evaluation()
