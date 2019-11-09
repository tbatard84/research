# Implementation of the three contrast processing methods used in section 4.1. of the paper... 

import argparse
import numpy as np
from scipy.fftpack import fft2, ifft2
from skimage.color import rgb2lab, lab2rgb
from scipy import signal
from PIL import Image

if __name__ == '__main__':
		parser = argparse.ArgumentParser(description='')
		parser.add_argument(
		'--input-image',
		help='image to process',
		action='store',
		dest='input_image',
		required=True)
		parser.add_argument(
		'--method',
		help='contrast processing method [channelwise, chrominance, huepreserving]',
		default='channelwise',
		action='store',
		dest='method',
		required=False)  
		parser.add_argument(
		'--enhancement-parameter',
		help='enhancement parameter (the model performs regularization if the parameter is negative)',
		type=float,
		default=0.45,
		action='store',
		dest='enhancement_param',
		required=False)
		parser.add_argument(
		'--std-kernel',
		help='standard deviation of the Gaussian kernel',
		type=float,
		default=3000.,
		action='store',
		dest='std',
		required=False)

		args = parser.parse_args()
		filename = args.input_image
		method = args.method
		gamma = args.enhancement_param
		std = args.std

		u0 = Image.open(filename)
		u0 = np.array(u0).astype(float)

		#Construction of the denominator
		gaussian_kernel_1d_1 = np.expand_dims(signal.gaussian(u0.shape[0],std),axis=1)
		gaussian_kernel_1d_2 = np.expand_dims(signal.gaussian(u0.shape[1],std),axis=1)
		gaussian_kernel_2d = np.matmul(gaussian_kernel_1d_1,np.transpose(gaussian_kernel_1d_2))
		gaussian_kernel_2d = gaussian_kernel_2d/np.sum(gaussian_kernel_2d)

		den = (1-gamma)*np.ones(u0.shape[:2]) + gamma*fft2(gaussian_kernel_2d) 


		v0 = rgb2lab(u0/255.)
		print(np.median(v0[:,:,0]))
		v = np.empty(u0.shape)

		if method == 'channelwise':

			for i in range(u0.shape[2]):
				v[:,:,i] = np.real(ifft2(fft2(v0[:,:,i])/den))

		elif method == 'chrominancepreserving':
			
			v = v0
			v[:,:,0] = np.real(ifft2(fft2(v0[:,:,0])/den))
	
		elif method == 'lightnesspreserving':

			v[:,:,0] = v0[:,:,0]
			v[:,:,1] = np.real(ifft2(fft2(v0[:,:,1])/den))
			v[:,:,2] = np.real(ifft2(fft2(v0[:,:,2])/den))	

		elif method == 'huepreserving':
		
			# from cartesian to spherical coordinates
			r0 = np.sqrt(np.square(v0[:,:,0]) + np.square(v0[:,:,1]) + np.square(v0[:,:,2]))
			theta0 = np.arccos(np.divide(v0[:,:,2],r0+0.001*np.ones(r0.shape)))
			phi0 = np.arctan(np.divide(v0[:,:,1],v0[:,:,0]+0.001*np.ones(r0.shape))) 

			r = np.fmax(np.zeros(r0.shape),np.real(ifft2(fft2(r0)/den)))

			# from spherical coordinates to cartesian coordinates
			v[:,:,0]  = r * np.sin(theta0) * np.cos(phi0)
			v[:,:,1]  = r * np.sin(theta0) * np.sin(phi0)
			v[:,:,2]  = r * np.cos(theta0)	

		else: raise NotImplementedError(f'Invalid contrast processing method')	

		print(np.median(v[:,:,0]))	
		u = 255*lab2rgb(v)

		u = np.clip(u, 0, 255)
		u = u.astype('uint8')
		img = Image.fromarray(u, 'RGB')
		img.save(f'res_{method}_method.png')	
