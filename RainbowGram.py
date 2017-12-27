'''
Created 12-26-17 by Matthew C. McCallum
'''

def RainbowGram( signal, samp_rate ):
	'''
	A quick and dirty function for plotting my personal version of the RainbowGram
	'''

	# Parameters
	win_len_secs = 0.02
	win_len = int( win_len_secs*samp_rate )
	overlap = 1.0 - 1.0/2.0/2.0/2.0
	frame_inc = int( ( 1 - overlap )*win_len )
	num_frames = math.floor( ( sig_len - win_len )/frame_inc ) + 1
	window = np.hamming( win_len )
	fft_size = int( win_len*2 )
	saturation = 0.9
	dyn_range = 60

	# Working variables
	expected_phase = np.arange( fft_size )/fft_size*2.0*np.pi*frame_inc

	# Create spectrogram
	frame_indices = np.arange( num_frames, dtype='int32' )
	freq_indices = np.arange( win_len, dtype='int32' )
	spec_indices = np.add( *np.meshgrid( frame_indices*frame_inc, freq_indices ) )
	spec = np.fft.fft( np.dot( np.diag( window ), signal[spec_indices] ), fft_size, axis=0 )

	# Get colors from phase
	phase_spec = np.angle( spec )
	phase_diff = ( ( phase_spec[:,1:] - phase_spec[:,:-1] ).T - expected_phase ).T
	phase_diff = phase_diff/np.pi
	frame_offset = ( np.around( phase_diff ) - phase_diff )*fft_size/2.0/frame_inc

	# Get image
	hue = frame_offset/max(np.max(np.max(frame_offset)),abs(np.min(np.min(frame_offset))))
	hue[hue>1.0] = 1.0
	hue[hue<-1.0] = -1.0
	hue = ( hue + 1.0 )/2.0
	sat = np.ones( hue.shape )*saturation
	val = 20*np.log10( np.abs( spec[:,1:] ) )
	val = val - np.max( np.max( val ) )
	val[val<-dyn_range] = -dyn_range
	val += dyn_range
	val /= dyn_range
	image = [[colorsys.hsv_to_rgb( h, s, v ) for h, s, v in zip( rowh, rows, rowv )] for rowh, rows, rowv in zip( hue, sat, val )]
	image = np.array( image )

	plt.imshow( image )
	plt.show()
