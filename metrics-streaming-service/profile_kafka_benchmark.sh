#!/bin/bash

# Profile Kafka Benchmark Script
# This script helps you run line-by-line profiling on the Kafka metrics streaming service

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CSV_FILE="./data/HDFS_v1/preprocessed/Event_traces.csv"
SKETCH_TYPE="quantileflow"
OUTPUT_DIR="./profiling_results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Kafka Metrics Streaming - Line Profiling Benchmark       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --csv)
            CSV_FILE="$2"
            shift 2
            ;;
        --sketch)
            SKETCH_TYPE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --csv FILE       CSV file path (default: ./data/HDFS_v1/preprocessed/Event_traces.csv)"
            echo "  --sketch TYPE    Sketch type: quantileflow, datadog, momentsketch, hdrhistogram (default: quantileflow)"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --sketch momentsketch"
            echo "  $0 --csv ./data/custom.csv --sketch quantileflow"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo -e "${YELLOW}⚠  Warning: CSV file not found: $CSV_FILE${NC}"
    echo "Please ensure the data file exists before running."
    exit 1
fi

# Check if line_profiler is installed
if ! python -c "import line_profiler" 2>/dev/null; then
    echo -e "${YELLOW}⚠  Warning: line_profiler is not installed${NC}"
    echo "Installing line_profiler..."
    pip install line_profiler
fi

# Generate timestamp for output file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/${SKETCH_TYPE}_profile_${TIMESTAMP}.txt"

echo -e "${GREEN}Configuration:${NC}"
echo "  • CSV File:      $CSV_FILE"
echo "  • Sketch Type:   $SKETCH_TYPE"
echo "  • Output File:   $OUTPUT_FILE"
echo ""

echo -e "${BLUE}Starting Kafka profiling benchmark...${NC}"
echo ""

# Check if Kafka is running
if ! lsof -i :9092 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠  Warning: Kafka doesn't appear to be running on port 9092${NC}"
    echo "Please start Kafka first. See README.md for instructions."
    echo ""
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run the profiling
python ./src/main.py \
    --csv "$CSV_FILE" \
    --sketch "$SKETCH_TYPE" \
    --line-profile \
    --profile-output "$OUTPUT_FILE"

echo ""
echo -e "${GREEN}✓ Profiling complete!${NC}"
echo ""
echo -e "${BLUE}Results saved to: ${OUTPUT_FILE}${NC}"
echo ""
echo "To view the results:"
echo "  cat $OUTPUT_FILE"
echo "  less $OUTPUT_FILE"
echo ""
echo "Tips for analyzing results:"
echo "  • Look for lines with high '% Time' - these are bottlenecks"
echo "  • Compare 'Time' vs 'Per Hit' to find frequently-called slow operations"
echo "  • Focus optimization on functions with high cumulative time"
