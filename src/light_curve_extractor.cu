

/*
  
  TODO: [Done]Add overlap frames to properly allow pixel delays
        [Done]Add support for per-source (and pixel) delays, just like weights
        Add interface for passing in and returning to device memory
        Test code
        CPU implementation
        Benchmarks
        [Done]Getter functions
        [Done]Documentation
        Further hardening
        [Done]Python wrapper
        C++ wrapper
        
*/

#include "light_curve_extractor.h"

#include <cstdio>
#include <cassert>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/remove.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>

// Only for testing
#include "stopwatch.hpp"

typedef unsigned char uchar;
texture<uchar,  cudaTextureType3D, cudaReadModeNormalizedFloat> t_data_8bit;
texture<ushort, cudaTextureType3D, cudaReadModeNormalizedFloat> t_data_16bit;
texture<float,  cudaTextureType3D, cudaReadModeElementType>     t_data_32bit;

// d_source_pixels contains the Cartesian coords (x,y,z) and weight (w) of
//   each pixel in each cell.
template<int BLOCK_SIZE, int BITDEPTH>
__global__
void extract_light_curves_kernel(uint                     nframes,
                                 uint                     nsources,
                                 const uint* __restrict__ c_source_offsets,
                                 const uint* __restrict__ c_source_npixels,
                                 const float4*            d_source_pixels,
                                 float*                   d_light_curves) {
	uint frame0 = threadIdx.x + blockIdx.x*BLOCK_SIZE;
	uint source = blockIdx.y;
	
	// Load constant values
	uint source_offset  = c_source_offsets[source];
	uint source_npixels = c_source_npixels[source];
	
	// Manually-managed shared memory cache for source pixel info
	__shared__ float4 s_source_pixels[BLOCK_SIZE];
	
	// Loop over whole grids of threads
	// Note: Must pad to multiple of block size due to use of smem/syncthreads
	uint nframes_padded = ((nframes - 1) / BLOCK_SIZE + 1) * BLOCK_SIZE;
	for( uint frame=frame0; frame<nframes_padded; frame+=BLOCK_SIZE*gridDim.x ) {
		// Sequentially sum up pixels contributing to this source
		float sum = 0.f;
		for( uint pb=0; pb<source_npixels; pb+=BLOCK_SIZE ) {
			// Take care of the last block potentially being smaller
			uint block_size = min(BLOCK_SIZE, source_npixels-pb);
			
			// Cache a line of pixel coords/weight for this source
			uint p = pb + threadIdx.x;
			if( threadIdx.x < block_size ) {
				s_source_pixels[threadIdx.x] = d_source_pixels[source_offset + p];
			}
			__syncthreads();
			// Sum pixels in the block
			for( uint pi=0; pi<block_size; ++pi ) {
				float4 pxl = s_source_pixels[pi];
				float  val;
				switch( BITDEPTH ) {
				case 8:
					val = tex3D(t_data_8bit,
					            pxl.x+.5f, pxl.y+.5f, pxl.z+.5f + frame);
					val *= (1<<8)-1; // Un-normalise back to integer scale
					break;
				case 16:
					val = tex3D(t_data_16bit,
					            pxl.x+.5f, pxl.y+.5f, pxl.z+.5f + frame);
					val *= (1<<16)-1; // Un-normalise back to integer scale
					break;
				case 32:
					val = tex3D(t_data_32bit,
					            pxl.x+.5f, pxl.y+.5f, pxl.z+.5f + frame);
					break;
				}
				sum += pxl.w * val;
			}
		}
		// Write the summed frame back to global mem
		// Note: It's now safe to crop excess threads
		if( frame < nframes ) {
			//d_light_curves[frame + nframes*source] = sum;
			// Note: Frame-major allows stiching frames together easily, but
			//         here means non-coalesced memory writes. It doesn't
			//         appear to make a significant difference to the run time.
			d_light_curves[source + nsources*frame] = sum;
		}
	}
}

#if defined(LCE_DEBUG) && LCE_DEBUG
#define throw_error(err) {	  \
	fprintf(stderr, "LCE error (%s:%i): %s\n", \
	        __FILE__, __LINE__, \
	        lce_get_error_string(err)); \
	return err; \
}
#define throw_cuda_error(cuda_err, err) {	  \
	fprintf(stderr, "LCE GPU error (%s:%i): %s\n", \
	        __FILE__, __LINE__, \
	        cudaGetErrorString(cuda_err)); \
	return err; \
}
#else
#define throw_error(err) return err
#define throw_cuda_error(cuda_err, err) return err
#endif

// Internal (tuning) parameters
enum {
	LCE_NBUF                   = 2,
	LCE_KERNEL_BLOCK_SIZE      = 128,
	LCE_DEFAULT_NFRAMES_DEVICE = 64
};

struct lce_plan_t {
	lce_size width;
	lce_size height;
	lce_size bitdepth;
	lce_size nframes_device;
	int      device_idx;
	lce_size overlap;
	thrust::device_vector<float> d_pixel_weights;
	thrust::device_vector<float> d_pixel_delays;
	cudaStream_t streams[LCE_NBUF];
	cudaArray*   a_data[LCE_NBUF];
	
	lce_size nsources;
	thrust::device_vector<uint>   d_source_offsets;
	thrust::device_vector<uint>   d_source_npixels;
	thrust::device_vector<float4> d_source_pixels;
	
	thrust::device_vector<float>  d_light_curves;
};

lce_error allocate_data_arrays(lce_plan plan) {
	cudaError_t cuda_error;
	for( int buf=0; buf<LCE_NBUF; ++buf ) {
		// Free existing allocation if present
		if( plan->a_data[buf] ) {
			cuda_error = cudaFreeArray(plan->a_data[buf]);
			if( cuda_error != cudaSuccess ) {
				throw_cuda_error(cuda_error, LCE_GPU_ERROR);
			}
		}
		
		// Allocate a 3D CUDA array for later binding to a texture
		cudaChannelFormatKind formatkind;
		switch( plan->bitdepth ) {
		case 8: 
		case 16: formatkind = cudaChannelFormatKindUnsigned; break;
		case 32: formatkind = cudaChannelFormatKindFloat; break;
		default: throw_error(LCE_INVALID_BITDEPTH);
		}
		cudaChannelFormatDesc channel_desc =
			cudaCreateChannelDesc(plan->bitdepth, 0, 0, 0,
			                      formatkind);
		cudaExtent   extent = make_cudaExtent(plan->width, plan->height,
		                                      plan->nframes_device
		                                      + plan->overlap);
		unsigned int flags  = 0;
		cuda_error = cudaMalloc3DArray(&plan->a_data[buf],
		                               &channel_desc,
		                               extent,
		                               flags);
		if( cuda_error != cudaSuccess ) {
			throw_cuda_error(cuda_error, LCE_MEM_ALLOC_FAILED);
		}
	}
	return LCE_NO_ERROR;
}

lce_size  lce_get_width(const lce_plan plan)  { return plan->width; }
lce_size  lce_get_height(const lce_plan plan) { return plan->height; }
lce_size  lce_get_bitdepth(const lce_plan plan) { return plan->bitdepth; }
lce_size  lce_get_nframes_device(const lce_plan plan) { return plan->nframes_device; }
int       lce_get_device_idx(const lce_plan plan) { return plan->device_idx; }
lce_size  lce_get_nsources(const lce_plan plan) { return plan->nsources; }
lce_size  lce_get_max_delay(const lce_plan plan) { return plan->overlap; }
lce_error lce_get_pixel_weights(const lce_plan plan,
                                float*         pixel_weights) {
	thrust::copy(plan->d_pixel_weights.begin(),
	             plan->d_pixel_weights.end(),
	             pixel_weights);
	
	return LCE_NO_ERROR;
}
lce_error lce_get_pixel_delays(const lce_plan plan,
                               float*         pixel_delays) {
	thrust::copy(plan->d_pixel_delays.begin(),
	             plan->d_pixel_delays.end(),
	             pixel_delays);
	
	return LCE_NO_ERROR;
}

lce_error lce_set_nframes_device(lce_plan plan,
                                 lce_size nframes_device) {
	if( nframes_device == 0 ) {
		throw_error(LCE_INVALID_NFRAMES);
	}
	plan->nframes_device = nframes_device;
	lce_error err = allocate_data_arrays(plan);
	if( err != LCE_NO_ERROR ) {
		throw_error(err);
	}
	return LCE_NO_ERROR;
}

lce_error lce_create(lce_plan*    plan,
                     lce_size     width,
                     lce_size     height,
                     lce_size     bitdepth,
                     int          device_idx,
                     const float* pixel_weights,
                     const float* pixel_delays) {
	/*
	printf("Width    = %u\n", width);
	printf("Height   = %u\n", height);
	printf("Bitdepth = %u\n", bitdepth);
	printf("Device   = %i\n", device_idx);
	printf("Weights  = %p\n", pixel_weights);
	printf("Delays   = %p\n", pixel_delays);
	*/
	if( !(bitdepth == 8 ||
	      bitdepth == 16 ||
	      bitdepth == 32) ) {
		throw_error(LCE_INVALID_BITDEPTH);
	}
	
	lce_plan newplan = new lce_plan_t;
	if( !newplan ) {
		throw_error(LCE_MEM_ALLOC_FAILED);
	}
	
	// TODO: Do careful clean-up of dynamic allocations in here when
	//         something fails mid-way through the function.
	
	newplan->width          = width;
	newplan->height         = height;
	newplan->bitdepth       = bitdepth;
	newplan->nframes_device = LCE_DEFAULT_NFRAMES_DEVICE;
	newplan->device_idx     = device_idx;
	newplan->nsources       = 0;
	
	cudaError_t cuda_error;
	cuda_error = cudaSetDevice(device_idx);
	if( cuda_error != cudaSuccess ) {
		if( cuda_error == cudaErrorInvalidDevice ) {
			throw_cuda_error(cuda_error, LCE_INVALID_DEVICE);
		}
		else {
			throw_cuda_error(cuda_error, LCE_GPU_ERROR);
		}
	}
	
	// TODO: Check the assign and resize calls for exceptions
	if( pixel_weights ) {
		newplan->d_pixel_weights.assign(pixel_weights,
		                                pixel_weights + width*height);
	}
	if( pixel_delays ) {
		newplan->d_pixel_delays.assign(pixel_delays,
		                               pixel_delays + width*height);
		float max_delay = *thrust::max_element(newplan->d_pixel_delays.begin(),
		                                       newplan->d_pixel_delays.end());
		newplan->overlap = ceil(max_delay);
	}
	else {
		newplan->overlap = 0;
	}
	
	for( int buf=0; buf<LCE_NBUF; ++buf ) {
		cudaStreamCreate(&newplan->streams[buf]);
		newplan->a_data[buf] = 0;
	}
	lce_error err = allocate_data_arrays(newplan);
	if( err != LCE_NO_ERROR ) {
		throw_error(err);
	}
	
	*plan = newplan;
	
	// Set textures to return 0 when out-of-bounds, and to use
	//   linear interpolation.
	t_data_8bit.addressMode[0] = cudaAddressModeBorder;
	t_data_8bit.addressMode[1] = cudaAddressModeBorder;
	t_data_8bit.addressMode[2] = cudaAddressModeBorder;
	t_data_8bit.filterMode  = cudaFilterModeLinear;
	t_data_8bit.normalized  = false;
	t_data_16bit.addressMode[0] = cudaAddressModeBorder;
	t_data_16bit.addressMode[1] = cudaAddressModeBorder;
	t_data_16bit.addressMode[2] = cudaAddressModeBorder;
	t_data_16bit.filterMode = cudaFilterModeLinear;
	t_data_16bit.normalized = false;
	t_data_32bit.addressMode[0] = cudaAddressModeBorder;
	t_data_32bit.addressMode[1] = cudaAddressModeBorder;
	t_data_32bit.addressMode[2] = cudaAddressModeBorder;
	t_data_32bit.filterMode = cudaFilterModeLinear;
	t_data_32bit.normalized = false;
	
	return LCE_NO_ERROR;
}
void lce_destroy(lce_plan plan) {
	if( !plan ) {
		return;
	}
	cudaSetDevice(plan->device_idx);
	for( int buf=0; buf<LCE_NBUF; ++buf ) {
		cudaStreamDestroy(plan->streams[buf]);
		cudaFreeArray(plan->a_data[buf]);
	}
	delete plan;
}
template<typename T>
struct abs_less_equal_val : public thrust::unary_function<T,bool> {
	T val;
	abs_less_equal_val(T val_) : val(val_) {}
	inline __host__ __device__
	bool operator()(T x) const {
		return fabs(x) <= val;
	}
};
template<typename T>
struct multiply_by : public thrust::unary_function<T,T> {
	T val;
	multiply_by(T val_) : val(val_) {}
	inline __host__ __device__
	T operator()(T x) const {
		return x * val;
	}
};
struct get_spatial_sort_index : public thrust::unary_function<uint, uint> {
	uint imsize;
	uint width;
	get_spatial_sort_index(uint imsize_, uint width_)
		: imsize(imsize_), width(width_) {}
	// This code was copied from http://graphics.stanford.edu/~seander/bithacks.html
	inline __host__ __device__
	uint get_zindex(uint x, uint y) const {
		const uint B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
		const uint S[] = {1, 2, 4, 8};
		// Interleave lower 16 bits of x and y, so the bits of x
		// are in the even positions and bits from y in the odd;
		// x and y must initially be less than 65536.
		x = (x | (x << S[3])) & B[3];
		x = (x | (x << S[2])) & B[2];
		x = (x | (x << S[1])) & B[1];
		x = (x | (x << S[0])) & B[0];
		
		y = (y | (y << S[3])) & B[3];
		y = (y | (y << S[2])) & B[2];
		y = (y | (y << S[1])) & B[1];
		y = (y | (y << S[0])) & B[0];
		
		uint z = x | (y << 1);
		return z;
	}
	inline __host__ __device__
	uint operator()(uint idx) const {
		uint src_idx = idx / imsize;
		uint pxl_idx = idx % imsize;
		uint x       = pxl_idx % width;
		uint y       = pxl_idx / width;
		uint zindex  = get_zindex(x, y);
		uint sort_index = zindex + imsize*src_idx;
		return sort_index;
	}
};
struct gen_source_pixel_table
	: public thrust::binary_function<uint, void, float4> {
	uint imsize;
	uint width;
	const float* pixel_delays;
	const float* pixel_weights;
	gen_source_pixel_table(uint imsize_, uint width_,
	                       const float* pixel_delays_,
	                       const float* pixel_weights_)
		: imsize(imsize_), width(width_),
		  pixel_delays(pixel_delays_),
		  pixel_weights(pixel_weights_) {}
	template<typename Tuple>
	inline __host__ __device__
	float4 operator()(uint idx, Tuple wd) const {
		float weight = thrust::get<0>(wd);
		float delay  = thrust::get<1>(wd);
		uint pxl_idx = idx % imsize;
		float4 pxl;
		pxl.x = pxl_idx % width;
		pxl.y = pxl_idx / width;
		pxl.z = delay  + pixel_delays[pxl_idx];
		pxl.w = weight * pixel_weights[pxl_idx];
		return pxl;
	}
};
template<typename T>
struct z_less_than : public thrust::binary_function<T,T,bool> {
	inline __host__ __device__
	bool operator()(T a, T b) const {
		return a.z < b.z;
	}
};
lce_error lce_set_source_weights_by_image(lce_plan     plan,
                                          lce_size     nsources,
                                          const float* source_weights,
                                          float        zero_thresh,
                                          const float* source_delays) {
	if( !plan ) {
		throw_error(LCE_INVALID_PLAN);
	}
	
	cudaSetDevice(plan->device_idx);
	
	plan->nsources = nsources;
	
	// TODO: It's possible that some applications will actually have few/no
	//         zeros in the source weights. In these cases, we could actually
	//         store the dense source_weights matrices and use sgemm instead
	//         of the custom 'sparse weights' kernel.
	//         Would have to guess or autotune the tipping point between the
	//           efficiency of the two algorithms.
	
	using thrust::make_counting_iterator;
	using thrust::make_transform_iterator;
	using thrust::make_zip_iterator;
	using thrust::make_tuple;
	
	// Copy weights to device
	size_t imsize = plan->width * plan->height;
	thrust::device_vector<float> d_weights(source_weights,
	                                       source_weights + nsources*imsize);
	// Copy delays to device
	thrust::device_vector<float> d_delays;
	if( !source_delays ) {
		d_delays.resize(nsources*imsize, 0.f);
	}
	else {
		d_delays.assign(source_delays,
		                source_delays + nsources*imsize);
	}
	
	// Compact a list of indices by removing zero weights
	thrust::device_vector<uint> d_inds(nsources*imsize);
	thrust::device_vector<uint>::iterator end_iter;
	end_iter = thrust::remove_copy_if(make_counting_iterator<uint>(0),
	                                  make_counting_iterator<uint>(nsources*imsize),
	                                  d_weights.begin(),
	                                  d_inds.begin(),
	                                  abs_less_equal_val<float>(zero_thresh));
	d_inds.resize(end_iter - d_inds.begin());
	
	// Find each source's offset into the compacted list of indices
	plan->d_source_offsets.resize(nsources);
	thrust::lower_bound(d_inds.begin(), d_inds.end(),
	                    make_transform_iterator(make_counting_iterator<uint>(0),
	                                            multiply_by<uint>(imsize)),
	                    make_transform_iterator(make_counting_iterator<uint>(nsources),
	                                            multiply_by<uint>(imsize)),
	                    plan->d_source_offsets.begin());
	
	// Difference adjacent offsets to find the number of pixels in each source
	plan->d_source_npixels.resize(nsources);
	plan->d_source_offsets.push_back(d_inds.size());
	thrust::transform(plan->d_source_offsets.begin()+1,
	                  plan->d_source_offsets.end(),
	                  plan->d_source_offsets.begin(),
	                  plan->d_source_npixels.begin(),
	                  thrust::minus<uint>());
	
	// Spatially sort inds (e.g., by Z order)
	// TODO: This has yet to prove its worth
	thrust::device_vector<uint> d_spatial_sort_keys(d_inds.size());
	thrust::transform(d_inds.begin(), d_inds.end(),
	                  d_spatial_sort_keys.begin(),
	                  get_spatial_sort_index(imsize, plan->width));
	thrust::sort_by_key(d_spatial_sort_keys.begin(),
	                    d_spatial_sort_keys.end(),
	                    d_inds.begin());
	
	if( plan->d_pixel_delays.empty() ) {
		plan->d_pixel_delays.resize(imsize, 0.f);
	}
	if( plan->d_pixel_weights.empty() ) {
		plan->d_pixel_weights.resize(imsize, 1.f);
	}
	
	// Generate source pixel lookup values as:
	//   float4(col, row, delay, weight*pixel_weight)
	const float* d_pixel_delays_ptr  = thrust::raw_pointer_cast(&plan->d_pixel_delays[0]);
	const float* d_pixel_weights_ptr = thrust::raw_pointer_cast(&plan->d_pixel_weights[0]);
	plan->d_source_pixels.resize(d_inds.size());
	thrust::transform(d_inds.begin(), d_inds.end(),
	                  make_permutation_iterator(make_zip_iterator(make_tuple(d_weights.begin(),
	                                                                         d_delays.begin())),
	                                            d_inds.begin()),
	                  plan->d_source_pixels.begin(),
	                  gen_source_pixel_table(imsize, plan->width,
	                                         d_pixel_delays_ptr,
	                                         d_pixel_weights_ptr));
	
	// Adjust required overlap based on actual max delay
	float4 max_val = *thrust::max_element(plan->d_source_pixels.begin(),
	                                      plan->d_source_pixels.end(),
	                                      z_less_than<float4>());
	float max_delay = max_val.z;
	plan->overlap = ceil(max_delay);
	
	// Check for illegal negative delays
	float4 min_val = *thrust::min_element(plan->d_source_pixels.begin(),
	                                      plan->d_source_pixels.end(),
	                                      z_less_than<float4>());
	float min_delay = min_val.z;
	if( min_delay < 0.f ) {
		throw_error(LCE_INVALID_DELAY);
	}
	
	// Allocate output memory space
	plan->d_light_curves.resize(nsources*plan->nframes_device);
	
	return LCE_NO_ERROR;
}
/*
// TODO: Implement this if there is a motivating use-case
int lce_set_source_weights_by_pixel(lce_plan        plan,
                                    lce_size        nsources,
                                    const lce_size* source_npixels,
                                    const int**     source_coords,
                                    const float**   source_weights) {
	
}
*/
lce_error copy_h2d(const lce_plan plan,
                   const void*    data,
                   int            buf) {
	assert(plan != 0);
	
	cudaError_t error;
	
	cudaStream_t stream = plan->streams[buf];
	
	error = cudaGetLastError();
	if( error != cudaSuccess ) {
		throw_cuda_error(error, LCE_MEM_COPY_FAILED);
	}
	
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr((void*)data,
	                                          plan->width*plan->bitdepth/8,
	                                          plan->width,
	                                          plan->height);
	copyParams.dstArray = plan->a_data[buf];
	copyParams.extent   = make_cudaExtent(plan->width,
	                                      plan->height,
	                                      plan->nframes_device + plan->overlap);
	copyParams.kind     = cudaMemcpyHostToDevice;
	error = cudaMemcpy3DAsync(&copyParams, stream);
	
	if( error != cudaSuccess ) {
		throw_cuda_error(error, LCE_MEM_COPY_FAILED);
	}
	return LCE_NO_ERROR;
}
lce_error compute(const lce_plan plan,
                  float*         light_curves,
                  int            buf) {
	assert(plan != 0);
	
	enum { BLOCK_SIZE = LCE_KERNEL_BLOCK_SIZE };
	
	cudaStream_t stream = plan->streams[buf];
	cudaError_t cuda_error;
	cudaArray* a_data = plan->a_data[buf];
	const lce_size* d_source_offsets_ptr = thrust::raw_pointer_cast(&plan->d_source_offsets[0]);
	const lce_size* d_source_npixels_ptr = thrust::raw_pointer_cast(&plan->d_source_npixels[0]);
	const float4*   d_source_pixels_ptr  = thrust::raw_pointer_cast(&plan->d_source_pixels[0]);
	float*          d_light_curves_ptr   = thrust::raw_pointer_cast(&plan->d_light_curves[0]);
	
	//Stopwatch timer;
	//timer.start();
	
	// TODO: Does this slow things down significantly? Could it be done during init, with separate textures for each buf?
	switch( plan->bitdepth ) {
	case 8:  cuda_error = cudaBindTextureToArray(t_data_8bit,  a_data); break;
	case 16: cuda_error = cudaBindTextureToArray(t_data_16bit, a_data); break;
	case 32: cuda_error = cudaBindTextureToArray(t_data_32bit, a_data); break;
	default: throw_error(LCE_INVALID_BITDEPTH);
	}
	if( cuda_error != cudaSuccess ) {
		throw_cuda_error(cuda_error, LCE_GPU_ERROR);
	}
	
	//timer.stop();
	//printf("Texture bind time = %f\n", timer.getTime());
	
	// Compute thread decomposition
	size_t nframe_blocks = (plan->nframes_device - 1) / BLOCK_SIZE + 1;
	dim3 block(BLOCK_SIZE);
	dim3 grid(nframe_blocks,
	          plan->nsources);
	// Dynamically dispatch on bitdepth and execute GPU kernel
	switch( plan->bitdepth ) {
	case 8:  extract_light_curves_kernel<BLOCK_SIZE,  8><<<grid, block, 0, stream>>>
			(plan->nframes_device,plan->nsources,d_source_offsets_ptr,
			 d_source_npixels_ptr,d_source_pixels_ptr,d_light_curves_ptr); break;
	case 16: extract_light_curves_kernel<BLOCK_SIZE, 16><<<grid, block, 0, stream>>>
			(plan->nframes_device,plan->nsources,d_source_offsets_ptr,
			 d_source_npixels_ptr,d_source_pixels_ptr,d_light_curves_ptr); break;
	case 32: extract_light_curves_kernel<BLOCK_SIZE, 32><<<grid, block, 0, stream>>>
			(plan->nframes_device,plan->nsources,d_source_offsets_ptr,
			 d_source_npixels_ptr,d_source_pixels_ptr,d_light_curves_ptr); break;
	default: throw_error(LCE_INVALID_BITDEPTH);
	}
	
#if defined(LCE_DEBUG) && LCE_DEBUG
	// Note: Error-checking the kernel disables asynchronous execution
	cudaStreamSynchronize(stream);
	cuda_error = cudaGetLastError();
	if( cuda_error != cudaSuccess ) {
		throw_cuda_error(cuda_error, LCE_GPU_ERROR);
	}
#endif
	
	// Copy results back to host
	size_t light_curve_nbytes = (plan->nframes_device * plan->nsources
	                             * sizeof(float));
	cuda_error = cudaMemcpyAsync((void*)light_curves, (void*)d_light_curves_ptr,
	                             light_curve_nbytes, cudaMemcpyDeviceToHost,
	                             stream);
	if( cuda_error != cudaSuccess ) {
		throw_cuda_error(cuda_error, LCE_MEM_COPY_FAILED);
	}
	
	return LCE_NO_ERROR;
}
lce_error lce_execute_async(const lce_plan plan,
                            lce_size       nframes,
                            const void*    data,
                            float*         light_curves) {
	if( !plan ) {
		throw_error(LCE_INVALID_PLAN);
	}
	
	lce_size nframes_computed = nframes - plan->overlap;
	
	if( nframes_computed % plan->nframes_device != 0 ) {
		throw_error(LCE_INVALID_NFRAMES);
	}
	
	cudaSetDevice(plan->device_idx);
	
	size_t npipe        = nframes_computed / plan->nframes_device;
	size_t frame_nbytes = plan->width*plan->height*plan->bitdepth/8;
	size_t in_stride    = plan->nframes_device * frame_nbytes;
	size_t out_stride   = plan->nframes_device * plan->nsources;
	
	size_t pipe = 0;
	copy_h2d(plan, (char*)data + pipe*in_stride, pipe % LCE_NBUF);
	while( pipe < npipe-1 ) {
		copy_h2d(plan, (char*)data + (pipe+1)*in_stride, (pipe+1) % LCE_NBUF);
		compute(plan, light_curves + pipe*out_stride, pipe % LCE_NBUF);
		++pipe;
	}
	compute(plan, light_curves + pipe*out_stride, pipe % LCE_NBUF);
	
	return LCE_NO_ERROR;
}
lce_error lce_execute(const lce_plan plan,
                      lce_size       nframes,
                      const void*    data,
                      float*         light_curves) {
	
	Stopwatch timer;
	timer.start();
	
	lce_error ret = lce_execute_async(plan, nframes, data, light_curves);
	if( ret != LCE_NO_ERROR ) {
		throw_error(ret);
	}
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if( error != cudaSuccess ) {
		throw_cuda_error(error, LCE_GPU_ERROR);
	}
	
	timer.stop();
	printf("lce_execute time = %f s\n", timer.getTime());
	printf("                 = %f fps\n", (nframes-plan->overlap)/timer.getTime());
	
	return ret;
}
lce_error lce_synchronize(const lce_plan plan) {
	cudaSetDevice(plan->device_idx);
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if( error != cudaSuccess ) {
		throw_cuda_error(error, LCE_GPU_ERROR);
	}
	return LCE_NO_ERROR;
}
const char* lce_get_error_string(lce_error error) {
	switch( error ) {
	case LCE_NO_ERROR:
		return "No error";
	case LCE_INVALID_PLAN:
		return "Invalid plan";
	case LCE_INVALID_BITDEPTH:
		return "Invalid bitdepth";
	case LCE_INVALID_NFRAMES:
		return "Invalid nframes";
	case LCE_INVALID_DEVICE:
		return "Invalid GPU device index";
	case LCE_INVALID_DELAY:
		return "Invalid delay";
	case LCE_MEM_ALLOC_FAILED:
		return "Memory allocation failed";
	case LCE_MEM_COPY_FAILED:
		return "Memory copy failed";
	case LCE_GPU_ERROR:
		return "GPU error";
	case LCE_UNKNOWN_ERROR:
		return "Unknown error. Please contact the author(s).";
	default:
		return "Invalid error code";
	}
}
lce_error lce_register_memory(void*    ptr,
                              lce_size size) {
	unsigned int flags = cudaHostRegisterPortable;
	cudaError_t error = cudaHostRegister(ptr, size, flags);
	if( error != cudaSuccess ) {
		throw_cuda_error(error, LCE_GPU_ERROR);
	}
	return LCE_NO_ERROR;
}
lce_error lce_unregister_memory(void* ptr) {
	cudaError_t error = cudaHostUnregister(ptr);
	if( error != cudaSuccess ) {
		throw_cuda_error(error, LCE_GPU_ERROR);
	}
	return LCE_NO_ERROR;
}
