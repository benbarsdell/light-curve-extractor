/*
 *  Copyright 2013 Ben Barsdell
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
  light_curve_extractor.h
  By Ben Barsdell (2013)
  benbarsdell@gmail.com
*/

/*! \file light_curve_extractor.h
 *  \brief Defines the interface to the light_curve_extractor library
 *
 * light_curve_extractor is a C library for extracting light curves from
 *   (pre-identified) sparse sources in image sequences. Computations are
 *   accelerated using NVIDIA GPUs via CUDA.
 */

#ifndef LCE_H_INCLUDE_GUARD
#define LCE_H_INCLUDE_GUARD

// Use C linkage to allow cross-language use of the library
#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned int       lce_size;
typedef struct lce_plan_t* lce_plan;

/*! \typedef lce_size
 * The size data-type used by the library to store sizes/dimensions. */
/*! \typedef lce_plan
 * The plan type used by the library to reference an lce computation plan.
     This is an opaque pointer type. */

typedef enum {
	LCE_NO_ERROR,
	LCE_NOT_READY,
	LCE_INVALID_PLAN,
	LCE_INVALID_BITDEPTH,
	LCE_INVALID_NFRAMES,
	LCE_INVALID_DEVICE,
	LCE_INVALID_DELAY,
	LCE_INVALID_FLAGS,
	LCE_MEM_ALLOC_FAILED,
	LCE_MEM_COPY_FAILED,
	LCE_GPU_ERROR,
	LCE_UNKNOWN_ERROR
} lce_error;

/*! \enum lce_error
 * Error codes for the library:\n
 * LCE_NO_ERROR: No error occurred.\n
 * LCE_INVALID_PLAN: The given plan is NULL.\n
 * LCE_INVALID_BITDEPTH: The specified bitdepth is unsupported; supported values are: 8 (uchar), 16 (ushort), 32 (float).\n
 * LCE_INVALID_NFRAMES: The given number of frames is 0 or not a multiple of the value returned by \p lce_get_nframes_device.\n
 * LCE_INVALID_DEVICE: The given GPU device index is invalid (no such device was detected in the system).\n
 * LCE_INVALID_DELAY: The total delay for a given source and pixel was negative; only positive delays are supported.\n
 * LCE_INVALID_FLAGS: The given flag(s) or combination of flags was invalid.\n
 * LCE_MEM_ALLOC_FAILED: A memory allocation failed.\n
 * LCE_MEM_COPY_FAILED: A memory copy failed.\n
 * LCE_GPU_ERROR: An error occurred relating to the GPU device.\n
 * LCE_UNKNOWN_ERROR: An unexpected error has occurred. Please contact the author(s) if you get this error.
 */

typedef enum {
	LCE_SYNC           = 1 << 0,
	LCE_ASYNC          = 1 << 1,
	LCE_SYNC_MASK      = LCE_SYNC | LCE_ASYNC,
	LCE_DEFAULT_SYNC   = LCE_SYNC,
	
	LCE_HOST_INPUT     = 1 << 2,
	LCE_DEVICE_INPUT   = 1 << 3,
	LCE_INPUT_MASK     = LCE_HOST_INPUT | LCE_DEVICE_INPUT,
	LCE_DEFAULT_INPUT  = LCE_HOST_INPUT,
	
	LCE_HOST_OUTPUT    = 1 << 4,
	LCE_DEVICE_OUTPUT  = 1 << 5,
	LCE_OUTPUT_MASK    = LCE_HOST_OUTPUT | LCE_DEVICE_OUTPUT,
	LCE_DEFAULT_OUTPUT = LCE_HOST_OUTPUT,
} lce_execute_flags;

/*! \enum lce_execute_flags
 * Execution flags for the library:\n
 * LCE_SYNC: Wait for the computation to finish before returning.\n
 * LCE_ASYNC: Do not wait for the computation to finish before returning.\n
 * LCE_HOST_INPUT: Input data passed to the library resides on the host.\n
 * LCE_DEVICE_INPUT: Input data passed to the library resides on the device.\n
 * LCE_HOST_OUTPUT: Output memory passed to the library resides on the host.\n
 * LCE_DEVICE_OUTPUT: Output memory passed to the library resides on the device.\n
 */

/*! \p lce_create builds a new plan object using the given parameters
 *  and returns it in \p *plan.
 *  
 *  \param plan           Pointer to an lce_plan object
 *  \param width          Width of the image data
 *  \param height         Height of the image data
 *  \param bitdepth       No. bits per pixel [must be 8 (uchar), 16 (ushort) or 32 (float)]
 *  \param device_idx     Index of the GPU device to use
 *  \param pixel_weights  (Optional) constant weight value for each of the width*height pixels
 *  \param pixel_delays   (Optional) constant frame delay for each of the
 *                          width*height pixels (note that linear interpolation
 *                          between frames is used when delays are fractional)
 *  \return One of the following error codes: \n
 *  \p LCE_NO_ERROR, \p LCE_INVALID_BITDEPTH,
 *  \p LCE_MEM_ALLOC_FAILED, \p LCE_MEM_COPY_FAILED,
 *  \p LCE_GPU_ERROR
 *  
 *  \note Optional parameters can be set to 0 to be ignored.
 */
lce_error lce_create(lce_plan*    plan,
                     lce_size     width,
                     lce_size     height,
                     lce_size     bitdepth,
                     int          device_idx,
                     const float* pixel_weights,
                     const float* pixel_delays);

/*! \p lce_destroy frees a plan and its associated resources
 *  
 *  \param plan Plan object to destroy
 */
void lce_destroy(lce_plan plan);

/*! \p lce_set_source_weights_by_image sets pixel weights for each source
 *       according to complete images (i.e., a weight for every image pixel)
 *  
 *  \param plan Plan object to set source weights for
 *  \param nsources No. sources in the data
 *  \param source_weights Weights for each source and pixel, ordered [source][row][column]
 *  \param zero_thresh Absolute weight threshold below which a pixel's
 *                       contribution to a source is ignored
 *  \param source_delays (Optional) frame delay for each source and pixel,
 *                         ordered [source][row][column]
 */
lce_error lce_set_source_weights_by_image(lce_plan     plan,
                                          lce_size     nsources,
                                          const float* source_weights,
                                          float        zero_thresh,
                                          const float* source_delays);
/*
int lce_set_source_weights_by_pixel(lce_plan        plan,
                                    lce_size        nsources,
                                    const lce_size* source_npixels,
                                    const int**     source_coords,
                                    const float**   source_weights,
                                    const float**   source_delays);
*/
/*! \p lce_execute executes a plan to extract light curves from the given data
 *  
 *  \param plan The plan to execute
 *  \param nframes No. frames in the image sequence. Note: Must be a multiple of
 *                   the value returned by \p lce_get_nframes_device \b after any
 *                   required overlap has been subtracted (see \ref lce_get_max_delay).
 *  \param data Pointer to the image sequence data, ordered [frame][row][column]
 *  \param light_curves Pointer to memory where output light curves will be placed, ordered [frame][source]
 *  \param flags (Optional) flags for plan execution. Combine flags with the '|' operator.
 *  \note The number of extraced frames in each light curve will be \p nframes - \p lce_get_max_delay(plan)
 */
lce_error lce_execute(const lce_plan plan,
                      lce_size       nframes,
                      const void*    data,
                      float*         light_curves,
                      unsigned int   flags);
/*! \p lce_execute_async asynchronously executes a plan to extract light curves from the given data\n
 *  See \ref lce_execute for details
 *  \note This function returns immediately, i.e., before the computation has finished.
 *          Call \p lce_synchronize to wait for execution to complete.
 */
/*
lce_error lce_execute_async(const lce_plan plan,
                            lce_size       nframes,
                            const void*    data,
                            float*         light_curves,
                            unsigned int   flags);
*/

/*! \p lce_query_status queries the status of plan execution, returning
 *       LCE_NO_ERROR if execution is finsihed, LCE_NOT_READY if execution
 *       is still in process, or another error code if an error occurred
 *       during plan execution.
 *
 *  \param plan The plan to query
 */
lce_error lce_query_status(const lce_plan plan);

/*! \p lce_synchronize waits for the asynchronous execution of a plan to complete before returning
 *
 *  \param plan The plan to synchronize
 */
lce_error lce_synchronize(const lce_plan plan);

lce_size  lce_get_width(const lce_plan plan);
lce_size  lce_get_height(const lce_plan plan);
lce_size  lce_get_bitdepth(const lce_plan plan);
/*! \p lce_get_nframes_device returns the number of frames processed on the
 *       device at a time. The number of frames passed to an \p lce_execute
 *       function must be a multiple of this value.
 *
 */
lce_size  lce_get_nframes_device(const lce_plan plan);
int       lce_get_device_idx(const lce_plan plan);
lce_size  lce_get_nsources(const lce_plan plan);
/*! \p lce_get_max_delay returns the plan's maximum pixel delay rounded up to
 *       a whole number of frames. This value represents the overlap required
 *       between adjacent sequences of frames, which must be included when
 *       passing data to the \p lce_execute* methods.
 */
lce_size  lce_get_max_delay(const lce_plan plan);
lce_error lce_get_pixel_weights(const lce_plan plan,
                                float*         pixel_weights);
lce_error lce_get_pixel_delays(const lce_plan plan,
                               float*         pixel_delays);
/*! \p lce_set_nframes_device sets the number of frames processed on the
 *       device at a time.
 *  \param nframes_device The number of frames to process on the device at a time
 *  \note This function must be called before calling any
 *          \p lce_set_source_weights* function.
 *  \note The \p nframes_device parameter impacts several things:\n
 *          a) The amount of memory used on the device.\n
 *          b) The efficiency of the computation on the device.\n
 *          c) The extent to which host-->device mem copies are overlapped
 *               with computation. Smaller values of \p nframes_device relative
 *               to the no. frames passed to \p lce_execute will result in
 *               greater overlap efficiency; however, making it too small will
 *               result in lower computational efficiency.
 *
 */
lce_error lce_set_nframes_device(lce_plan plan,
                                 lce_size nframes_device);

const char* lce_get_error_string(lce_error error);

/*! \p lce_register_memory can be used to 'register' memory spaces that are
 *       used for input to and output from the library. Registering has the
 *       effect of page-locking the memory, allowing fast and asynchronous
 *       memory copies to and from the device.
 *
 *  \param ptr Pointer to the (host) memory to register
 *  \param size Size in bytes of the memory
 */
lce_error lce_register_memory(void*    ptr,
                              lce_size size);

/*! \p lce_unregister_memory unregisters memory registered with \p lce_register_memory
 *
 *  \param ptr Pointer to the (host) memory to unregister
 */
lce_error lce_unregister_memory(void* ptr);

#ifdef __cplusplus
} // closing brace for extern "C"
#endif

#endif // LCE_H_INCLUDE_GUARD
