#!/bin/bash

# Build and run NCNN Deconvolution1D comparison test
# Usage: ./run_test.sh [options]

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check if we're on ARM architecture
    ARCH=$(uname -m)
    print_status "Architecture: $ARCH"
    
    if [[ "$ARCH" == "aarch64" || "$ARCH" == "armv7l" ]]; then
        print_success "ARM architecture detected - NEON optimizations available"
    else
        print_warning "Non-ARM architecture - NEON optimizations will be disabled"
    fi
    
    # Check for compiler
    if ! command -v g++ &> /dev/null; then
        print_error "g++ compiler not found. Please install build-essential."
        exit 1
    fi
    
    # Check for make
    if ! command -v make &> /dev/null; then
        print_error "make not found. Please install build-essential."
        exit 1
    fi
    
    print_success "All requirements satisfied"
}

# Function to clean build artifacts
clean_build() {
    print_status "Cleaning previous build artifacts..."
    make clean 2>/dev/null || true
    print_success "Build artifacts cleaned"
}

# Function to build the test program
build_test() {
    print_status "Building test program..."
    
    # Show compiler information
    make info
    
    # Build the program
    if make; then
        print_success "Build completed successfully"
    else
        print_error "Build failed"
        exit 1
    fi
}

# Function to run the test
run_test() {
    print_status "Running deconvolution comparison test..."
    echo ""
    
    if ./test_deconv1d_simple; then
        echo ""
        print_success "All tests passed!"
    else
        echo ""
        print_error "Some tests failed!"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -c, --clean    Clean build artifacts and exit"
    echo "  -b, --build    Build only (don't run tests)"
    echo "  -r, --run      Run tests only (assume already built)"
    echo "  -d, --debug    Build with debug information"
    echo "  -v, --verbose  Show verbose output"
    echo ""
    echo "Examples:"
    echo "  $0              # Clean, build, and run tests"
    echo "  $0 --clean      # Clean build artifacts"
    echo "  $0 --build      # Build only"
    echo "  $0 --debug      # Build with debug info and run"
}

# Parse command line arguments
CLEAN_ONLY=false
BUILD_ONLY=false
RUN_ONLY=false
DEBUG_BUILD=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -c|--clean)
            CLEAN_ONLY=true
            shift
            ;;
        -b|--build)
            BUILD_ONLY=true
            shift
            ;;
        -r|--run)
            RUN_ONLY=true
            shift
            ;;
        -d|--debug)
            DEBUG_BUILD=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
print_status "NCNN Deconvolution1D Comparison Test Runner"
print_status "============================================"

# Check requirements first
check_requirements

# Handle clean-only option
if [ "$CLEAN_ONLY" = true ]; then
    clean_build
    print_success "Clean completed"
    exit 0
fi

# Handle run-only option
if [ "$RUN_ONLY" = true ]; then
    if [ ! -f "./test_deconv1d_simple" ]; then
        print_error "Test executable not found. Please build first."
        exit 1
    fi
    run_test
    exit 0
fi

# Normal flow: clean, build, and optionally run
clean_build

if [ "$DEBUG_BUILD" = true ]; then
    print_status "Building with debug information..."
    make debug
else
    build_test
fi

# Handle build-only option
if [ "$BUILD_ONLY" = true ]; then
    print_success "Build completed"
    exit 0
fi

# Run the tests
run_test

print_success "Test execution completed successfully!"
