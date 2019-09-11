import numpy as np
from numpy import mean, sqrt, square, arange
from scipy.io import wavfile

# A.1
# how about the last block drop out?
def block_audio(x,blockSize,hopSize,fs):
	xb = []
	timeInSec = []
	frames = len(x)
	idx = 0

	while frames > idx:
		xb.append(x[:blockSize])
		timeInSec.append(idx)
		idx += hopSize

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