# The definitions in this file should be changed to match your system

CUDA_PATH ?= /usr/local/cuda
# CUDA_PATH ?= /opt/local/cuda

CUDA_DIR   ?= $(CUDA_PATH)
THRUST_DIR ?= $(CUDA_DIR)/include

SYSTEM_INC_DIR ?= /usr/local/include
SYSTEM_LIB_DIR ?= /usr/local/lib

# LIB_ARCH   = lib
LIB_ARCH   = lib64
# GPU_ARCH   = sm_13
# GPU_ARCH   = sm_20
GPU_ARCH   = sm_30

GCC        = gcc
GXX        = g++
AR         = ar
NVCC       = $(CUDA_DIR)/bin/nvcc -Xptxas -abi=no
DOXYGEN    = doxygen
RM         = rm
ECHO       = echo
MKFLAGS    = 

LCE_DEBUG = 0

#.SILENT :
