#!/usr/bin/env python
"""
Wrapper and test application for liblce.so
By Ben Barsdell (2013)
"""

import numpy
import array
from ctypes import *

# TODO: Should this instead reference an installed location (e.g., /usr/local/lib ?)
lib = cdll.LoadLibrary('lib/liblce.so')

lib.lce_get_width.restype          = c_uint
lib.lce_get_height.restype         = c_uint
lib.lce_get_bitdepth.restype       = c_uint
lib.lce_get_nframes_device.restype = c_uint
lib.lce_get_device_idx.restype     = c_int
lib.lce_get_nsources.restype       = c_uint
lib.lce_get_error_string.restype   = c_char_p

class LCE_PLAN_STRUCT(Structure):
	pass
LCE_PLAN = POINTER(LCE_PLAN_STRUCT)

class LightCurveExtractor(object):
	LCE_NO_ERROR = 0
	def _check_error(self, ret):
		if ret != self.LCE_NO_ERROR:
			raise(Exception(self.get_error_string(ret)))
	def __init__(self,
	             width, height, bitdepth,
	             device_idx=0,
	             pixel_weights=None, pixel_delays=None):
		self.lib = lib
		self.obj = LCE_PLAN()
		#self._light_curves = numpy.zeros((1,1), dtype=numpy.float32)
		
		if pixel_weights is None:
			pixel_weights_ptr = 0
		else:
			pixel_weights = numpy.array(pixel_weights, dtype=numpy.float32)
			pixel_weights_ptr = pixel_weights.ctypes.data_as(POINTER(c_float))
		if pixel_delays is None:
			pixel_delays_ptr = 0
		else:
			pixel_delays = numpy.array(pixel_delays, dtype=numpy.float32)
			pixel_delays_ptr = pixel_delays.ctypes.data_as(POINTER(c_float))
		
		ret = self.lib.lce_create(pointer(self.obj),
		                          width, height, bitdepth,
		                          device_idx,
		                          pixel_weights_ptr, pixel_delays_ptr)
		self._check_error(ret)
	def __del__(self):
		self.lib.lce_destroy(self.obj)
	def set_source_weights_by_image(self, nsources, source_weights,
	                                zero_thresh=1e-6, source_delays=None):
		source_weights = numpy.array(source_weights, dtype=numpy.float32)
		source_weights_ptr = source_weights.ctypes.data_as(POINTER(c_float))
		if source_delays is None:
			source_delays_ptr = 0
		else:
			source_delays = numpy.array(source_delays, dtype=numpy.float32)
			source_delays_ptr = source_delays.ctypes.data_as(POINTER(c_float))
		ret = self.lib.lce_set_source_weights_by_image(self.obj,
		                                               nsources,
		                                               source_weights_ptr,
		                                               c_float(zero_thresh),
		                                               source_delays_ptr)
		self._check_error(ret)
	def execute(self, nframes, data, light_curves=None):
		allocated_data = False
		if not isinstance(data, numpy.ndarray) or data.dtype != self.dtype:
			allocated_data = True
			data = numpy.array(data, dtype=self.dtype)
			self.register_memory(data)
		data_ptr = data.ctypes.data_as(POINTER(self.ctype))
		
		allocated_light_curves = False
		if light_curves is None:
			allocated_light_curves = True
			light_curves = numpy.zeros((nframes,self.nsources), dtype=numpy.float32)
			self.register_memory(light_curves)
		light_curves_ptr = light_curves.ctypes.data_as(POINTER(c_float))
		
		ret = self.lib.lce_execute(self.obj,
		                           nframes,
		                           data_ptr,
		                           light_curves_ptr)
		self._check_error(ret)
		
		if allocated_light_curves:
			self.unregister_memory(light_curves)
		if allocated_data:
			self.unregister_memory(data)
			
		return light_curves
	
	def execute_async(self, nframes, data, light_curves=None):
		allocated_data = False
		if not isinstance(data, numpy.ndarray) or data.dtype != self.dtype:
			allocated_data = True
			data = numpy.array(data, dtype=self.dtype)
			self.register_memory(data)
		data_ptr = data.ctypes.data_as(POINTER(self.ctype))
		
		self._light_curves = light_curves
		allocated_light_curves = False
		if self._light_curves is None:
			allocated_light_curves = True
			self._light_curves = numpy.zeros((nframes,self.nsources), dtype=numpy.float32)
			self.register_memory(self._light_curves)
			
		light_curves_ptr = self._light_curves.ctypes.data_as(POINTER(c_float))
		
		ret = self.lib.lce_execute_async(self.obj,
		                                 nframes,
		                                 data_ptr,
		                                 light_curves_ptr)
		self._check_error(ret)
		
		if allocated_light_curves:
			self.unregister_memory(self._light_curves)
		if allocated_data:
			self.unregister_memory(data)
		
	def synchronize(self):
		ret = self.lib.lce_synchronize(self.obj)
		self._check_error(ret)
		return self._light_curves
	@property
	def width(self):
		return self.lib.lce_get_width(self.obj)
	@property
	def height(self):
		return self.lib.lce_get_height(self.obj)
	@property
	def bitdepth(self):
		return self.lib.lce_get_bitdepth(self.obj)
	@property
	def dtype(self):
		if self.bitdepth == 8:
			return numpy.uint8
		elif self.bitdepth == 16:
			return numpy.uint16
		elif self.bitdepth == 32:
			return numpy.float32
		else:
			raise ValueError("Invalid bitdepth")
	@property
	def ctype(self):
		if self.bitdepth == 8:
			return c_uint8
		elif self.bitdepth == 16:
			return c_uint16
		elif self.bitdepth == 32:
			return c_float
		else:
			raise ValueError("Invalid bitdepth")
	@property
	def nframes_device(self):
		return self.lib.lce_get_nframes_device(self.obj)
	@property
	def device_idx(self):
		return self.lib.lce_get_device_idx(self.obj)
	@property
	def nsources(self):
		return self.lib.lce_get_nsources(self.obj)
	@property
	def max_delay(self):
		return self.lib.lce_get_max_delay(self.obj)
	def get_pixel_weights(self):
		pixel_weights = numpy.zeros((self.height,self.width), dtype=numpy.float32)
		pixel_weights_ptr = pixel_weights.ctypes.data_as(POINTER(c_float))
		ret = self.lib.lce_get_pixel_weights(self.obj, pixel_weights_ptr)
		self._check_error(ret)
		return pixel_weights
	def get_pixel_delays(self):
		pixel_delays = numpy.zeros((self.height,self.width), dtype=numpy.float32)
		pixel_delays_ptr = pixel_delays.ctypes.data_as(POINTER(c_float))
		ret = self.lib.lce_get_pixel_delays(self.obj, pixel_delays_ptr)
		self._check_error(ret)
		return pixel_delays
	def get_error_string(self, err):
		return self.lib.lce_get_error_string(err)
	def register_memory(self, mem_array):
		ptr = mem_array.ctypes.data_as(POINTER(c_char))
		size = mem_array.size*mem_array.dtype.itemsize
		ret = self.lib.lce_register_memory(ptr, size)
		self._check_error(ret)
	def unregister_memory(self, mem_array):
		ptr = mem_array.ctypes.data_as(POINTER(c_char))
		ret = self.lib.lce_unregister_memory(ptr)
		self._check_error(ret)

if __name__ == "__main__":
	import numpy as np
	import time
	import matplotlib.pyplot as plt
	
	width          = 2000
	height         = 200
	bitdepth       = 16
	nframes        = 1024
	device_idx     = 0
	pixel_weights  = np.ones((height, width)) * 1.0
	pixel_delays   = np.zeros((height, width)) + 0.31415
	
	print "Creating plan"
	lce = LightCurveExtractor(width, height, bitdepth,
	                          device_idx,
	                          #None, None)
	                          pixel_weights, pixel_delays)
	
	print "Generating source weights"
	nsources = 32
	np.random.seed(1234);
	source_weights = np.zeros((nsources, height, width))
	# Create a mask for each source as a randomly-placed circular region
	i = np.arange(width)
	j = np.arange(height)[:,np.newaxis]
	for s in xrange(nsources):
		x = np.random.random()*width
		y = np.random.random()*height
		dist = np.sqrt((x-i)**2 + (y-j)**2)
		source_weights[s] = dist < np.sqrt(width*height)/np.sqrt(nsources)/2*4/np.pi
	
	print "Setting source weights"
	lce.set_source_weights_by_image(nsources, source_weights)
	
	print "Generating data"
	overlap = lce.max_delay
	nframes_o = nframes + overlap
	# Note: Setting the exact data type is only necessary if wanting to
	#         avoid re-allocations when calling the execute functions.
	if bitdepth == 8:
		data = np.ones((nframes_o, height, width), dtype=np.uint8) * 1
	elif bitdepth == 16:
		data = np.ones((nframes_o, height, width), dtype=np.uint16) * 1
	else:
		data = np.ones((nframes_o, height, width), dtype=np.float32) * 1
	# Add linear scaling over frames
	data *= np.linspace(1, nframes_o, nframes_o)[:,np.newaxis,np.newaxis]
	print "Executing plan"
	tstart = time.time()
	
	lce.register_memory(data)
	
	light_curves = np.zeros((nframes, nsources), dtype=np.float32)
	lce.register_memory(light_curves)
	
	light_curves = lce.execute(nframes_o, data, light_curves)
	#lce.execute(nframes_o, data, light_curves)
	#lce.execute_async(nframes_o, data, light_curves)
	#light_curves = lce.synchronize()
	#lce.synchronize()
	elapsed = time.time() - tstart
	print "Elapsed =", elapsed
	
	lce.unregister_memory(data)
	
	print "Light curves:"
	print light_curves
	
	print "Plotting..."
	plt.plot(light_curves)
	plt.show()
	
	print "Done"
