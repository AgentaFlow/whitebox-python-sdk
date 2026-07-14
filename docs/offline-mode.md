# Offline Mode

WhiteBoxXAI SDK supports offline mode for robust operation when network connectivity is unreliable.

## Overview

Offline mode provides:
- **Persistent Queue**: SQLite-based storage that survives application restarts
- **Auto-Sync**: Background synchronization every 60 seconds (configurable)
- **Priority-Based**: Operations are synced based on priority (CRITICAL > HIGH > NORMAL > LOW)
- **Retry Logic**: Automatic retry with exponential backoff (max 3 attempts)
- **Thread-Safe**: Supports concurrent operations safely

## Quick Start

```python
from whiteboxxai import WhiteBoxXAI

# Enable offline mode with auto-sync
client = WhiteBoxXAI(
    api_key="your-api-key",
    enable_offline=True,
    offline_dir="./whiteboxxai_offline",
    offline_auto_sync=True,
    offline_sync_interval=60
)
```

## Configuration

### Basic Configuration

```python
client = WhiteBoxXAI(
    api_key="your-api-key",
    enable_offline=True,              # Enable offline mode
    offline_dir="./offline_queue",    # Storage directory
    offline_max_queue_size=10000,     # Max queued operations (0 = unlimited)
)
```

### Auto-Sync Configuration

```python
client = WhiteBoxXAI(
    api_key="your-api-key",
    enable_offline=True,
    offline_auto_sync=True,           # Enable background sync
    offline_sync_interval=60,         # Sync every 60 seconds
)
```

## Managing the Queue

### Check Queue Status

```python
status = client.get_offline_status()
print(f"Queue size: {status['queue_size']}")
print(f"Last sync: {status['last_sync_time']}")
print(f"Failed operations: {status['failed_count']}")
```

### Manual Sync

```python
result = client.sync_offline_queue()
print(f"Synced: {result['synced']}")
print(f"Failed: {result['failed']}")
```

### Cleanup Old Operations

```python
# Remove operations older than 7 days
client.cleanup_offline_queue(older_than_days=7)
```

## Priority Levels

Set priority for operations:

```python
# HIGH priority prediction
monitor.log_prediction(
    inputs={"feature1": 1.0},
    output={"prediction": 0.85},
    priority="HIGH"
)

# CRITICAL priority for important events
monitor.log_prediction(
    inputs={"feature1": 1.0},
    output={"prediction": 0.95},
    priority="CRITICAL"
)
```

Priority levels: `CRITICAL`, `HIGH`, `NORMAL` (default), `LOW`

## Best Practices

1. **Storage Location**: Choose a persistent location for `offline_dir`
2. **Queue Size**: Set appropriate `offline_max_queue_size` based on your storage
3. **Sync Interval**: Balance between freshness and API load
4. **Cleanup**: Regularly cleanup old operations to prevent queue bloat
5. **Monitoring**: Monitor queue size and failed operations

## Error Handling

Offline mode handles errors gracefully:

```python
try:
    monitor.log_prediction(inputs=data, output=prediction)
except Exception as e:
    # Operation is automatically queued if offline mode is enabled
    print(f"Logged to offline queue: {e}")
```

## Production Deployment

For production deployments:

```python
client = WhiteBoxXAI(
    api_key=os.getenv("WHITEBOXAI_API_KEY"),
    enable_offline=True,
    offline_dir="/var/lib/whiteboxxai/queue",
    offline_max_queue_size=50000,
    offline_auto_sync=True,
    offline_sync_interval=30,  # More frequent sync in production
)

# Monitor queue health
import schedule

def check_queue_health():
    status = client.get_offline_status()
    if status['queue_size'] > 10000:
        alert_ops_team(f"Queue size: {status['queue_size']}")

schedule.every(5).minutes.do(check_queue_health)
```

## See Also

- [Getting Started](getting-started.md)
- [API Reference](api-reference.md)
- [Production Deployment](PRODUCTION_DEPLOYMENT.md)
