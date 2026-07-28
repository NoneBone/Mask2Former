BUILD ?= release
CUDA_INSTALL_DIR ?= /usr/local/cuda
TRT_DIR ?= $(TENSORRT_DIR)
TRT_INC_DIR ?= $(TRT_DIR)/include
TRT_LIB_DIR ?= $(TRT_DIR)/lib
TRT_OSS_DIR ?= /root/cys/PROJECT/00-COMMON/DEMO/02-TensorRT

CXX := g++
NVCC := $(CUDA_INSTALL_DIR)/bin/nvcc
TARGET := libpose_ms_deform_attn_plugin.so
BUILD_DIR := build

COMMON_INC := $(TRT_OSS_DIR)/plugin
INCLUDES := -I. -I$(TRT_INC_DIR) -I$(CUDA_INSTALL_DIR)/include -I$(COMMON_INC)
LIBS := -L$(TRT_LIB_DIR) -L$(CUDA_INSTALL_DIR)/lib64 -lnvinfer -lcudart
RPATH := -Wl,-rpath,$(TRT_LIB_DIR) -Wl,-rpath,$(CUDA_INSTALL_DIR)/lib64

CXXFLAGS := -std=c++17 -fPIC -Wall -Wextra $(INCLUDES)
NVCCFLAGS := -std=c++17 -Xcompiler -fPIC $(INCLUDES) \
             -gencode=arch=compute_86,code=sm_86 \
             -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
ifeq ($(BUILD),release)
CXXFLAGS += -O3 -DNDEBUG
NVCCFLAGS += -O3
else
CXXFLAGS += -O0 -g
NVCCFLAGS += -O0 -g -lineinfo
endif

OBJS := $(BUILD_DIR)/multiscaleDeformableAttnPluginLegacy.o \
        $(BUILD_DIR)/multiscaleDeformableAttn.o \
        $(BUILD_DIR)/PluginRegistration.o \
        $(BUILD_DIR)/PluginLogShim.o

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) -shared -o $@ $(OBJS) $(LIBS) $(RPATH)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
