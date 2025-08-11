#!/bin/bash

# Simple compilation script for the deconvolution comparison test
# This script compiles the simplified version without complex dependencies

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== NCNN Deconvolution1D Comparison Test Builder ===${NC}"

# Check architecture
ARCH=$(uname -m)
echo -e "${BLUE}Architecture: ${ARCH}${NC}"

# Compiler settings
CXX=g++
CXXFLAGS="-std=c++11 -O2 -Wall -Wextra"

# ARM-specific flags
if [[ "$ARCH" == "aarch64" ]]; then
    CXXFLAGS="$CXXFLAGS -march=armv8-a -mtune=cortex-a72"
    echo -e "${GREEN}ARM64 optimizations enabled${NC}"
elif [[ "$ARCH" == "armv7l" ]]; then
    CXXFLAGS="$CXXFLAGS -march=armv7-a -mfpu=neon -mtune=cortex-a9"
    echo -e "${GREEN}ARMv7 NEON optimizations enabled${NC}"
fi

# Check if ARM NEON is available
if [[ "$ARCH" == *"arm"* ]] || [[ "$ARCH" == "aarch64" ]]; then
    CXXFLAGS="$CXXFLAGS -D__ARM_NEON"
    echo -e "${GREEN}ARM NEON support enabled${NC}"
else
    echo -e "${YELLOW}ARM NEON support disabled (not ARM architecture)${NC}"
fi

# Libraries
LIBS="-lm -lpthread"

# Check for OpenMP support
if $CXX -fopenmp -x c++ -E /dev/null >/dev/null 2>&1; then
    CXXFLAGS="$CXXFLAGS -fopenmp"
    LIBS="$LIBS -lgomp"
    echo -e "${GREEN}OpenMP support enabled${NC}"
else
    echo -e "${YELLOW}OpenMP support not available${NC}"
fi

# Clean previous build
echo -e "${BLUE}Cleaning previous build...${NC}"
rm -f test_deconv1d_simple

# Compile
echo -e "${BLUE}Compiling test program...${NC}"
echo "Command: $CXX $CXXFLAGS -o test_deconv1d_simple test_deconv1d_simple.cpp $LIBS"

if $CXX $CXXFLAGS -o test_deconv1d_simple test_deconv1d_simple.cpp $LIBS; then
    echo -e "${GREEN}✓ Compilation successful!${NC}"
    
    # Make executable
    chmod +x test_deconv1d_simple
    
    echo -e "${BLUE}Running test...${NC}"
    echo "================================"
    
    # Run the test
    if ./test_deconv1d_simple; then
        echo "================================"
        echo -e "${GREEN}✓ All tests completed successfully!${NC}"
    else
        echo "================================"
        echo -e "${RED}✗ Some tests failed!${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Compilation failed!${NC}"
    exit 1
fi
