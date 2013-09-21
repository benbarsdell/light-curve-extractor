
include Makefile.inc

# Output directories
BIN_DIR     = bin
OBJ_DIR     = obj
LIB_DIR     = lib
INCLUDE_DIR = include

SRC_DIR   := src
#INC_DIR   := ./include
OPTIMISE  := -O3
DEBUG     := -g -DLCE_DEBUG=$(LCE_DEBUG)

INCLUDE   := -I$(SRC_DIR) -I$(THRUST_DIR)
LIB       := -L$(CUDA_DIR)/$(LIB_ARCH) -lcudart

SOURCES   := $(SRC_DIR)/light_curve_extractor.cu
HEADERS   := $(SRC_DIR)/light_curve_extractor.h
INTERFACE := $(SRC_DIR)/light_curve_extractor.h
#CPP_INTERFACE := $(SRC_DIR)/LCEPlan.hpp

LIB_NAME  := liblce
SO_EXT    := .so
A_EXT     := .a
MAJOR     := 1
MINOR     := 0.1
SO_FILE   := $(LIB_NAME)$(SO_EXT).$(MAJOR).$(MINOR)
SO_NAME   := $(LIB_DIR)/$(SO_FILE)
A_NAME    := $(LIB_DIR)/$(LIB_NAME)$(A_EXT)E
VERSION_FILE := liblce.version

PTX_NAME  := ./lce_kernels.ptx

all: shared

#$(ECHO) Building shared library $(SO_FILE)
shared: $(SO_NAME)

$(SO_NAME): $(SOURCES) $(HEADERS)
	$(NVCC) -c -Xcompiler "-fPIC -Wall" $(OPTIMISE) $(DEBUG) -arch=$(GPU_ARCH) $(INCLUDE) -o $(OBJ_DIR)/light_curve_extractor.o $(SRC_DIR)/light_curve_extractor.cu
	$(GCC) -shared -Wl,--version-script=$(VERSION_FILE),-soname,$(LIB_NAME)$(SO_EXT).$(MAJOR) -o $(SO_NAME) $(OBJ_DIR)/light_curve_extractor.o $(LIB)
	ln -s -f $(SO_FILE) $(LIB_DIR)/$(LIB_NAME)$(SO_EXT).$(MAJOR)
	ln -s -f $(SO_FILE) $(LIB_DIR)/$(LIB_NAME)$(SO_EXT)
	cp $(INTERFACE) $(INCLUDE_DIR)
#cp $(CPP_INTERFACE) $(INCLUDE_DIR)

#static: $(A_NAME)

#$(A_NAME): $(SRC_DIR)/dedisp.cu $(HEADERS)
#	$(NVCC) -c -Xcompiler "-fPIC -Wall" -arch=$(GPU_ARCH) $(OPTIMISE) $(DEBUG) -o $(OBJ_DIR)/dedisp.o $(SRC_DIR)/dedisp.cu
#	$(AR) rcs $(A_NAME) $(OBJ_DIR)/dedisp.o
#	cp $(INTERFACE) $(INCLUDE_DIR)
#	cp $(CPP_INTERFACE) $(INCLUDE_DIR)

test: $(SO_NAME)
	cd test; $(MAKE) $(MKARGS)

ptx: $(PTX_NAME)

$(PTX_NAME): $(SOURCES) $(LIB_DIR)/liblce.so $(HEADERS)
	$(NVCC) -ptx -Xcompiler "-fPIC -Wall" $(OPTIMISE) $(DEBUG) -arch=$(GPU_ARCH) $(INCLUDE) -o $(PTX_NAME) $(SRC_DIR)/lce.cu

install: $(SO_NAME)
	cp $(INTERFACE)                             $(SYSTEM_INC_DIR)
	cp $(SO_FILE)                               $(SYSTEM_LIB_DIR)
	cp $(LIB_DIR)/$(LIB_NAME)$(SO_EXT).$(MAJOR) $(SYSTEM_LIB_DIR)
	cp $(LIB_DIR)/$(LIB_NAME)$(SO_EXT)          $(SYSTEM_LIB_DIR)

doc: $(SRC_DIR)/light_curve_extractor.h Doxyfile
	$(DOXYGEN) Doxyfile

clean:
	$(RM) -f $(SO_NAME) $(A_NAME) $(OBJ_DIR)/*.o
