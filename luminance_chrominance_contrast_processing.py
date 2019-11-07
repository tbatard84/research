import argparse
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
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
		'--parameter-enhancement-luminance',
		help='enhancement parameter of the luminance component (the model performs regularization if the parameter is negative)',
		type=float,
		default=0.,
		action='store',
		dest='param_luminance',
		required=False)
		parser.add_argument(
		'--parameter-enhancement-chrominance',
		help='enhancement parameter of the chrominance component (the model performs regularization if the parameter is negative)',
		type=float,
		default=0.45,
		action='store',
		dest='param_chrominance',
		required=False)
		parser.add_argument(
		'--std-kernel-luminance',
		help='standard deviation of the Gaussian kernel used to process the luminance component',
		type=float,
		default=3000.,
		action='store',
		dest='std_luminance',
		required=False)
		parser.add_argument(
		'--std-kernel-chrominance',
		help='standard deviation of the Gaussian kernel used to process the chrominance component',
		type=float,
		default=3000.,
		action='store',
		dest='std_chrominance',
		required=False)

		args = parser.parse_args()
		filename = args.input_image
		gamma_l = args.param_luminance
		gamma_c = args.param_chrominance
		std_l = args.std_luminance
		std_c = args.std_chrominance

		u0 = Image.open(filename)
		u0 = np.array(u0).astype(float)	

		# From RGB coordinates to Luminance-Chrominance coordinates
		v0 = np.empty(u0.shape)
		v0[:,:,0] = (u0[:,:,0]+u0[:,:,1]+u0[:,:,2])/np.sqrt(3.)
		v0[:,:,1] = (-u0[:,:,1]/np.sqrt(2.) + u0[:,:,2]/np.sqrt(2.))
		v0[:,:,2] = (u0[:,:,0] - u0[:,:,1]/2. - u0[:,:,2]/2.)/np.sqrt(3/2)
 
		
		#Construction of the denominators in expression ()
		gaussian_kernel_1d_1 = np.expand_dims(signal.gaussian(u0.shape[0],std_l),axis=1)
		gaussian_kernel_1d_2 = np.expand_dims(signal.gaussian(u0.shape[1],std_l),axis=1)
		gaussian_kernel_2d = np.matmul(gaussian_kernel_1d_1,np.transpose(gaussian_kernel_1d_2))
		gaussian_kernel_2d = gaussian_kernel_2d/np.sum(gaussian_kernel_2d)
		den_l = fft2(gaussian_kernel_2d)*gamma_l + np.ones(u0.shape[:2])*(1- gamma_l)

		# Computation of expressions
		gaussian_kernel_1d_1 = np.expand_dims(signal.gaussian(u0.shape[0],std_c),axis=1)
		gaussian_kernel_1d_2 = np.expand_dims(signal.gaussian(u0.shape[1],std_c),axis=1)
		gaussian_kernel_2d = np.matmul(gaussian_kernel_1d_1,np.transpose(gaussian_kernel_1d_2))
		gaussian_kernel_2d = gaussian_kernel_2d/np.sum(gaussian_kernel_2d)
		den_c = fft2(gaussian_kernel_2d)*gamma_c + np.ones(u0.shape[:2])*(1- gamma_c)

		v = np.empty(u0.shape)
		v[:,:,0] = np.real(ifft2(fft2(v0[:,:,0])/den_l))
		v[:,:,1] = np.real(ifft2(fft2(v0[:,:,1])/den_c))
		v[:,:,2] = np.real(ifft2(fft2(v0[:,:,2])/den_c))

		# From Luminance-Chrominance to RGB coordinates
		u = np.empty(u0.shape)
		u[:,:,0] = (v[:,:,0]/np.sqrt(3.)+v[:,:,2]*np.sqrt(2./3.))
		u[:,:,1] = (v[:,:,0]/np.sqrt(3.)-np.sqrt(2)*v[:,:,2]/(2.*np.sqrt(3.))-v[:,:,1]/np.sqrt(2.))
		u[:,:,2] = (v[:,:,0]/np.sqrt(3.)-np.sqrt(2)*v[:,:,2]/(2.*np.sqrt(3.))+v[:,:,1]/np.sqrt(2.))

		u = np.clip(u, 0, 255)
		u = u.astype('uint8')
		img = Image.fromarray(u, 'RGB')
		img.save('res.png')