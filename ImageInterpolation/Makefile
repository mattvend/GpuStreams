################################################################################
#
# Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to NVIDIA ownership rights under U.S. and
# international Copyright laws.
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.  This source code is a "commercial item" as
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda-8.0

##############################
# start deprecated interface #
##############################
ifeq ($(x86_64),1)
	$(info WARNING - x86_64 variable has been deprecated)
	$(info WARNING - please use TARGET_ARCH=x86_64 instead)
	TARGET_ARCH ?= x86_64
endif
ifneq ($(GCC),)
	$(info WARNING - GCC variable has been deprecated)
	$(info WARNING - please use HOST_COMPILER=$(GCC) instead)
	HOST_COMPILER ?= $(GCC)
endif
ifneq ($(abi),)
	$(error ERROR - abi variable has been removed)
endif
############################
# end deprecated interface #
############################

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le armv7l))
	ifneq ($(TARGET_ARCH),$(HOST_ARCH))
		ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
			TARGET_SIZE := 64
		else ifneq (,$(filter $(TARGET_ARCH),armv7l))
			TARGET_SIZE := 32
		endif
	else
		TARGET_SIZE := $(shell getconf LONG_BIT)
	endif
else
	$(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
	ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
		$(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
	endif
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
	$(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -D_FORCE_INLINES -m${TARGET_SIZE} -Wno-deprecated-gpu-targets -std=c++11
CCFLAGS     := -Wno-write-strings -Wno-delete-incomplete -std=c++11 
LDFLAGS     :=

# build flags
ifeq ($(TARGET_OS),darwin)
	LDFLAGS += -rpath $(CUDA_PATH)/lib
	CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
	LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
	CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
	LDFLAGS += -pie
	CCFLAGS += -fpie -fpic -fexceptions
endif

# Debug build flags
ifeq ($(dbg),1)
	  NVCCFLAGS += -g -G
	  BUILD_TYPE := debug
else
	  BUILD_TYPE := release
endif

TOP := $(dir $(lastword $(MAKEFILE_LIST)))

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

SAMPLE_ENABLED := 1

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I$(TOP) -I../../common/inc
LIBRARIES :=

################################################################################

# Gencode arguments
SMS ?= 20 30 35 37 50 52 60

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################

# Target rules
all: build

build: ImCpu.o ImGpu.o ImGpu2.o ImGpus.o ImGpu2s.o MainStream.o MainStream MainThread.o MainThread Interpolate Main.o

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

# Building cpu class
ImCpu.o:ImCpu.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

# Building gpu class without streams
ImGpu.o:ImGpu.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

ImGpu2.o:ImGpu.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

MainThread.o:MainThread.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

MainThread: MainThread.o ImCpu.o ImGpu.o ImGpu2.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) cp $@ ../Release

Main.o:Main.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

Interpolate: Main.o ImCpu.o ImGpu.o ImGpu2.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) cp $@ ../Release

# Building gpu class with streams
ImGpus.o:ImGpu.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -D USE_STREAMS $(GENCODE_FLAGS) -o $@ -c $<

ImGpu2s.o:ImGpu.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -D USE_STREAMS $(GENCODE_FLAGS) -o $@ -c $<

MainStream.o:MainThread.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -D USE_STREAMS $(GENCODE_FLAGS) -o $@ -c $<

MainStream: MainStream.o ImCpu.o ImGpus.o ImGpu2s.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) cp $@ ../Release

run: build
	$(EXEC) ./Main

clean:
	rm -f Interpolate Main Main.o MainThread MainThread.o MainStream MainStream.o ImCpu.o ImGpu.o ImGpu2.o ImGpus.o ImGpu2s.o

clobber: clean
