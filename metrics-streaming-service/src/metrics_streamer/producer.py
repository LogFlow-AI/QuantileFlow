from kafka import KafkaProducer
import pandas as pd
import json
import logging
from .config import KAFKA_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def serialize_message(x):
    return json.dumps(x).encode('utf-8')

def on_success(record_metadata):
    logger.debug(f"Message sent to {record_metadata.topic}[{record_metadata.partition}] @ offset {record_metadata.offset}")

def on_error(exc):
    logger.error(f"Error sending message: {exc}")

class LogProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
            client_id=f"{KAFKA_CONFIG['client_id']}-producer",
            value_serializer=serialize_message,
            acks='all',
            retries=5,
            retry_backoff_ms=1000
        )

    def send_metric(self, data):
        future = self.producer.send(
            KAFKA_CONFIG['topic'], 
            value=data
        ).add_callback(on_success).add_errback(on_error)
        return future

    def stream_logs(self, csv_file: str):
        try:
            # Print CSV columns for debugging
            df = pd.read_csv(csv_file, nrows=1)
            logger.info(f"CSV columns: {df.columns.tolist()}")
            
            for chunk in pd.read_csv(csv_file, chunksize=100):
                for _, row in chunk.iterrows():
                    data = {
                        'block_id': row['BlockId'],
                        'latency': int(row['Latency'])
                    }
                    self.send_metric(data)
                    logger.debug(f"Sent metric: {data}")
        except Exception as e:
            logger.error(f"Error streaming logs: {e}")
            raise
        finally:
            self.producer.flush()
            self.producer.close()
