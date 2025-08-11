# Makefile for NCNN Deconvolution1D comparison test

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Wextra

# ARM-specific flags
ifeq ($(shell uname -m),aarch64)
    CXXFLAGS += -march=armv8-a -mtune=cortex-a72
endif

ifeq ($(shell uname -m),armv7l)
    CXXFLAGS += -march=armv7-a -mfpu=neon -mtune=cortex-a9
endif

# Check if ARM NEON is available
ifneq (,$(findstring arm,$(shell uname -m)))
    CXXFLAGS += -D__ARM_NEON
endif

# Include directories
INCLUDES = -I../src -I../src/layer -I../src/layer/arm

# Libraries
LIBS = -lm -lpthread

# OpenMP support
ifeq ($(shell $(CXX) -fopenmp -x c++ -E /dev/null >/dev/null 2>&1; echo $$?),0)
    CXXFLAGS += -fopenmp
    LIBS += -lgomp
endif

# Target
TARGET = test_deconv1d_simple
SOURCE = test_deconv1d_simple.cpp

# Build rules
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)

# Run the test
test: $(TARGET)
	./$(TARGET)

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

# Show compiler info
info:
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS)"
	@echo "Includes: $(INCLUDES)"
	@echo "Libraries: $(LIBS)"
	@echo "Architecture: $(shell uname -m)"
