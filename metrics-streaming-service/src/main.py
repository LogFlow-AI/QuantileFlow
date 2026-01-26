import argparse
import logging
import sys
from multiprocessing import Process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_producer(csv_file):
    from metrics_streamer.producer import LogProducer
    producer = LogProducer()
    producer.stream_logs(csv_file)

def run_consumer(sketch_type, enable_line_profile=False, profile_output='line_profile_kafka.txt'):
    from metrics_streamer.consumer import LatencyMonitor
    consumer = LatencyMonitor(
        sketch_type=sketch_type,
        enable_line_profile=enable_line_profile,
        profile_output=profile_output
    )
    consumer.process_metrics()

def main():
    parser = argparse.ArgumentParser(
        description='Metrics Streaming Service with Sketch Algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with QuantileFlow DDSketch
  python main.py --csv data/HDFS_v1/HDFS_v1.csv
  
  # Compare different sketch algorithms
  python main.py --csv data/HDFS_v1/HDFS_v1.csv --sketch momentsketch
  
  # Enable line-by-line profiling for performance analysis
  python main.py --csv data/HDFS_v1/HDFS_v1.csv --line-profile
  
  # Profile with custom output file
  python main.py --csv data/HDFS_v1/HDFS_v1.csv --line-profile --profile-output results/profile.txt
  
  # Profile different sketch implementation
  python main.py --csv data/HDFS_v1/HDFS_v1.csv --sketch momentsketch --line-profile --profile-output results/momentsketch_profile.txt
        """
    )
    parser.add_argument('--csv', required=True, help='Input CSV file path')
    parser.add_argument('--sketch', choices=['quantileflow', 'datadog', 'momentsketch', 'hdrhistogram'], 
                        default='quantileflow', help='Sketch algorithm to use (default: quantileflow)')
    parser.add_argument('--line-profile', action='store_true',
                        help='Enable line-by-line profiling of sketch operations (requires line_profiler)')
    parser.add_argument('--profile-output', type=str, default='line_profile_kafka.txt',
                        help='Output file for line profiling results (default: line_profile_kafka.txt)')
    args = parser.parse_args()

    try:
        # Create processes with target functions
        consumer_process = Process(
            target=run_consumer, 
            args=(args.sketch, args.line_profile, args.profile_output)
        )
        producer_process = Process(target=run_producer, args=(args.csv,))

        # Start processes
        consumer_process.start()
        producer_process.start()

        # Wait for completion
        producer_process.join()
        consumer_process.join()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        producer_process.terminate()
        consumer_process.terminate()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
