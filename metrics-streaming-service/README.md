# Metrics Streaming Service

This is a metrics streaming service that allows us to stream and aggregate metrics effectively!

## Developer's Guide

### 1. Install Kafka (KRaft Mode - No Zookeeper Required)

Kafka 4.x uses KRaft mode, which eliminates the need for Zookeeper.

```bash
# Install Kafka
brew install kafka

# Note: Zookeeper is NOT needed for Kafka 4.x
```

### 2. Install Python Dependencies

```bash
# Activate your virtual environment if needed, then:
pip install -r requirements.txt
```

### 3. Initialize Kafka (KRaft Mode) and Run the Application

```bash
# 1. Generate a cluster UUID and format storage (first time only)
KAFKA_CLUSTER_ID=$(kafka-storage random-uuid)
kafka-storage format -t $KAFKA_CLUSTER_ID -c /opt/homebrew/etc/kafka/kraft/server.properties

# 2. Start Kafka with KRaft config
kafka-server-start /opt/homebrew/etc/kafka/kraft/server.properties &

# 3. Wait for Kafka to start (~10 seconds), then verify it's running
sleep 10
lsof -i :9092

# 4. Delete stale kafka topics (if any)
kafka-topics --delete --bootstrap-server localhost:9092 --topic latency-metrics 2>/dev/null || true

# 5. Create topic with replication-factor 1 for local testing
kafka-topics --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic latency-metrics

# 6. Verify topic creation
kafka-topics --list --bootstrap-server localhost:9092

# 7. Run application (see Sketch Options below)
python ./src/main.py --csv ./data/HDFS_v1/preprocessed/Event_traces.csv --sketch quantileflow
```

### 4. Sketch Algorithm Options

You can choose between different sketch algorithms using the `--sketch` flag:

```bash
# QuantileFlow DDSketch (default)
python ./src/main.py --csv ./data/HDFS_v1/preprocessed/Event_traces.csv --sketch quantileflow

# Datadog DDSketch
python ./src/main.py --csv ./data/HDFS_v1/preprocessed/Event_traces.csv --sketch datadog

# MomentSketch
python ./src/main.py --csv ./data/HDFS_v1/preprocessed/Event_traces.csv --sketch momentsketch

# HDRHistogram
python ./src/main.py --csv ./data/HDFS_v1/preprocessed/Event_traces.csv --sketch hdrhistogram
```

| Flag Value | Sketch Implementation | Description |
|------------|----------------------|-------------|
| `quantileflow` | QuantileFlow DDSketch | Custom DDSketch implementation |
| `datadog` | Datadog DDSketch | Official Datadog DDSketch library |
| `momentsketch` | QuantileFlow MomentSketch | Moment-based quantile estimation |
| `hdrhistogram` | QuantileFlow HDRHistogram | High Dynamic Range Histogram |

### 5. Monitor & Observe

```bash
# Terminal 2: Monitor consumer group
watch -n 1 kafka-consumer-groups --bootstrap-server localhost:9092 \
  --describe --group latency-monitor-group

# Terminal 3: Monitor topic metrics
watch -n 1 kafka-run-class kafka.tools.GetOffsetShell \
  --broker-list localhost:9092 --topic latency-metrics
```

### 6. Stopping Kafka

```bash
# Stop Kafka server
kafka-server-stop
```

---

## Data Setup

Place your HDFS v1 dataset in the following location:

```
data/HDFS_v1/preprocessed/Event_traces.csv
```

The CSV file should have the following columns:
- `BlockId` - Block identifier
- `Latency` - Latency value in milliseconds

---

## Example Output

Below is an example of the output from the latency monitor dashboard. This was taken from a run of the application on the HDFS_v1 dataset with parallelization enabled (num_workers = 4, partitions = 4).

```
================================================================================
Latency Monitor Dashboard - 2025-05-06 13:31:02
Sketch Algorithm: QuantileFlow DDSketch
================================================================================

Block ID: blk_-9128742458709757181
Current Latency:   131.00 ms

Performance Metrics:
Messages Processed:   575061
Data Points in Sketch:   575061
Runtime:   462.02 sec
Throughput:  1244.65 msg/sec
Refresh Rate:     0.00 sec

Latency Statistics:
  Mean:     0.00 ms

QuantileFlow DDSketch Percentiles:
  P50:   7260.81 ms
  P95:  50529.42 ms
  P99:  53654.09 ms
================================================================================
```

---

## Troubleshooting

### Kafka won't start / Connection refused on port 9092

1. **Check if Kafka is running:**
   ```bash
   lsof -i :9092
   ```

2. **If using `brew services` and getting errors:** Kafka 4.x requires KRaft mode. Use manual start instead:
   ```bash
   kafka-server-start /opt/homebrew/etc/kafka/kraft/server.properties
   ```

3. **Re-format storage if needed:**
   ```bash
   # Stop Kafka first, then:
   rm -rf /opt/homebrew/var/lib/kraft-combined-logs
   KAFKA_CLUSTER_ID=$(kafka-storage random-uuid)
   kafka-storage format -t $KAFKA_CLUSTER_ID -c /opt/homebrew/etc/kafka/kraft/server.properties
   ```

### Intel Mac Users

Replace `/opt/homebrew/` with `/usr/local/` in all paths above.
