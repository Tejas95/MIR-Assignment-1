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
    ACF = np.correlate(inputVector, inputVector, mode='full')
    normFactor= 1 / sqrt(2 * sum(square(inputVector)))
    if bIsNormalized == True:
        r=ACF[ACF.size // 2:]*normFactor
    else:
        r=ACF[ACF.size // 2:]
    return r

# A.3
def get_f0_from_acf(r, fs):
    peaks, _ = find_peaks(r)
    sort1 = np.flip(np.sort(r[peaks]),axis = 0)
    if len(sort1)>=2:
        for i in range (len(r)):
            if sort1[0] == r[i]:
                x1 = i
            
            if sort1[1] == r[i]:
                x2 = i

        firstpeak = x1
        secondpeak = x2

        period = secondpeak - firstpeak
        f0 = np.float(fs/period)
        return f0;
   
# A.4
def track_pitch_acf(x,blockSize,hopSize,fs):
    xb, timeInSec = block_audio(x,blockSize,hopSize,fs)
	
    NumOfBlocks   = int(np.round(len(x)/hopSize))

    f0 = np.zeros((NumOfBlocks))

    for i in range (len(xb)):
        r = comp_acf(xb[i], True)
        f0[i] = get_f0_from_acf(r,fs)  

    return f0, timeInSec;

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
    err_nonzero=[]
    err_sec=[]
    
    for i in range(len(f0)):
        if 0<=timeInSec[i]<1:
            err[i]=f0[i]-441
        elif timeInSec[i]>=1:
            err[i]=f0[i]-882
        if err[i] >0 or err[i] <0:
            err_nonzero=np.append(err[i],err_nonzero)
            err_sec=np.append([timeInSec[i],timeInSec[i]+1024/44100],err_sec)
  
    #plot f0 
    
    plt.plot(timeInSec,f0)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Estimated Frequency (Hz)')
    plt.title('Figure 1. Estimated Frequency of Test Signal')
    plt.show
    
    plt.show(block=False)
    
    #plot error
    
    plt.plot(timeInSec,err)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Difference between Estimated and Actual Frequency (Hz)')
    plt.title('Figure 2. Error of Estimated Test Signal Frequency')
    plt.show
    
    plt.show(block=False)
    #output error and location of the block in seconds for blocks with nonzero error
    return err_nonzero, err_sec
   
#sinusoidal_test()

# B.2 
def convert_freq2midi(freqInHz):
  
    typ=type(freqInHz)

    #convert frequency to MIDI pitch for a scalar
    if freqInHz <= 0:
        pitchInMIDI = 0
    
    else:
        pitchInMIDI=round(69+12*np.log2(freqInHz/440))
   
    #do the same thing for a vector 
    #pitch function converts a single item, then np.vectorize applies it to an array
    
    return pitchInMIDI

# B.3
def eval_pitchtrack(estimateInHz, groundtruthInHz):

    length = min(len(estimateInHz),len(groundtruthInHz))

    print (estimateInHz)
    estimateInHz = np.array([convert_freq2midi(f) for f in estimateInHz]) *100
    groundtruthInHz = np.array([convert_freq2midi(f) for f in groundtruthInHz]) *100
   
    non_zero = (groundtruthInHz>0)

    error = groundtruthInHz[non_zero] - estimateInHz[non_zero]

    rms = np.zeros(len(error))
    for i in range (len(error)):
        rms[i] = square(error[i])
    mean1 = mean(rms)
    rms1 = sqrt(mean1)
    return rms1

# B.4
def run_evaluation():

    blockSize = 1024
    hopSize = 512
    fs = 44100

    def read_label(path, estimateTime):

        es_idx = 0
        pre = -1

        oup = []
        time = []
        f = open(path, "r")
        for x in f:
            time = float(x.split('     ')[0])
            if es_idx < len(estimateTime):
                while es_idx < len(estimateTime) and estimateTime[es_idx] < time and estimateTime[es_idx] > pre:
                    oup.append(x.split('     ')[2])
                    pre = estimateTime[es_idx]
                    es_idx+=1
        return oup

    files = ['01-D_AMairena','24-M1_AMairena-Martinete','63-M2_AMairena']
    for file in files:
        fs, wav = wavfile.read('./trainData/'+file+'.wav')

        f0, timeInSec = track_pitch_acf(wav, blockSize, hopSize, fs)
        
        gtHz = np.array(read_label('./trainData/'+file+'.f0.Corrected.txt',timeInSec)).astype(np.float)
        rms = eval_pitchtrack(f0, gtHz) 
        print(rms)

run_evaluation()
