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

def run_consumer(sketch_type):
    from metrics_streamer.consumer import LatencyMonitor
    consumer = LatencyMonitor(sketch_type=sketch_type)
    consumer.process_metrics()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Input CSV file path')
    parser.add_argument('--sketch', choices=['quantileflow', 'datadog', 'momentsketch', 'hdrhistogram'], 
                        default='quantileflow', help='Sketch algorithm to use (default: quantileflow)')
    args = parser.parse_args()

    try:
        # Create processes with target functions
        consumer_process = Process(target=run_consumer, args=(args.sketch,))
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
