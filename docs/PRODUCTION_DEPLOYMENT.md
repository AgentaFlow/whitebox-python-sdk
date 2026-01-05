# Production Deployment Guide

This guide covers best practices for deploying WhiteBoxAI monitoring in production environments.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Environment Setup](#environment-setup)
3. [Configuration Management](#configuration-management)
4. [Performance Optimization](#performance-optimization)
5. [Security Best Practices](#security-best-practices)
6. [High Availability](#high-availability)
7. [Monitoring the Monitor](#monitoring-the-monitor)
8. [Disaster Recovery](#disaster-recovery)
9. [Scaling Strategies](#scaling-strategies)

---

## Architecture Overview

### Recommended Production Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ ML Service 1 │  │ ML Service 2 │  │ ML Service N │      │
│  │ + WhiteBoxAI │  │ + WhiteBoxAI │  │ + WhiteBoxAI │      │
│  │   SDK        │  │   SDK        │  │   SDK        │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Load Balancer  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  WhiteBoxAI API │
                    │   (FastAPI)     │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
   ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
   │  PostgreSQL │   │    Redis    │   │   S3/Blob   │
   │   Primary   │   │    Cache    │   │   Storage   │
   └─────────────┘   └─────────────┘   └─────────────┘
```

### Components

1. **SDK Integration**: Lightweight SDK embedded in ML services
2. **API Gateway**: Load-balanced WhiteBoxAI API endpoints
3. **Data Layer**: PostgreSQL + TimescaleDB, Redis cache, object storage
4. **Worker Layer**: Background workers for explanations and reports

---

## Environment Setup

### Production Environment Variables

```bash
# API Configuration
WHITEBOXAI_API_KEY=prod-api-key-here
WHITEBOXAI_BASE_URL=https://api.whiteboxai.yourcompany.com
WHITEBOXAI_ENVIRONMENT=production

# Performance Tuning
WHITEBOXAI_BATCH_SIZE=100
WHITEBOXAI_ASYNC_LOGGING=true
WHITEBOXAI_SAMPLING_RATE=0.1
WHITEBOXAI_CACHE_ENABLED=true
WHITEBOXAI_CACHE_TTL=3600

# Reliability
WHITEBOXAI_MAX_RETRIES=3
WHITEBOXAI_RETRY_BACKOFF=exponential
WHITEBOXAI_TIMEOUT=30
WHITEBOXAI_CIRCUIT_BREAKER_ENABLED=true

# Security
WHITEBOXAI_TLS_VERIFY=true
WHITEBOXAI_PII_DETECTION=true
WHITEBOXAI_DATA_MASKING=true

# Monitoring
WHITEBOXAI_ENABLE_METRICS=true
WHITEBOXAI_METRICS_PORT=9090
WHITEBOXAI_LOG_LEVEL=info
```

### Configuration File (config.yaml)

```yaml
whiteboxai:
  # API Settings
  api:
    base_url: ${WHITEBOXAI_BASE_URL}
    api_key: ${WHITEBOXAI_API_KEY}
    timeout: 30
    max_retries: 3

  # Logging Configuration
  logging:
    enabled: true
    level: info
    async: true
    batch_size: 100
    flush_interval: 60  # seconds

  # Sampling Strategy
  sampling:
    rate: 0.1  # Log 10% of predictions
    strategy: random  # random, stratified, or priority
    priority_threshold: 0.8  # For priority sampling

  # Caching
  cache:
    enabled: true
    backend: redis
    ttl: 3600
    max_size: 10000

  # Explanations
  explanations:
    enabled: true
    auto_generate: false
    sample_rate: 0.01
    method: shap
    cache_results: true

  # Drift Detection
  drift:
    enabled: true
    check_frequency: hourly
    window_size: 1000
    methods: [psi, ks_test]

  # Security
  security:
    pii_detection: true
    data_masking: true
    encrypt_at_rest: true
    tls_verify: true
```

---

## Configuration Management

### Using Environment-Specific Configs

```python
import os
from whiteboxai import WhiteBoxAI
from whiteboxai.config import Config

# Load config based on environment
env = os.getenv('ENVIRONMENT', 'development')
config = Config.from_file(f'config.{env}.yaml')

# Initialize client with config
client = WhiteBoxAI(config=config)
```

### Secrets Management

#### AWS Secrets Manager

```python
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name):
    client = boto3.client('secretsmanager', region_name='us-east-1')
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return response['SecretString']
    except ClientError as e:
        raise e

# Get API key from secrets manager
api_key = get_secret('whiteboxai/api-key')
client = WhiteBoxAI(api_key=api_key)
```

#### HashiCorp Vault

```python
import hvac

def get_vault_secret(path):
    client = hvac.Client(url='https://vault.yourcompany.com')
    client.token = os.getenv('VAULT_TOKEN')
    secret = client.secrets.kv.v2.read_secret_version(path=path)
    return secret['data']['data']

# Get API key from Vault
secrets = get_vault_secret('whiteboxai/production')
client = WhiteBoxAI(api_key=secrets['api_key'])
```

---

## Performance Optimization

### 1. Batching

Always use batch logging for production:

```python
from whiteboxai.monitor import ModelMonitor
from collections import deque
import threading
import time

class BatchedMonitor:
    def __init__(self, monitor, batch_size=100, flush_interval=60):
        self.monitor = monitor
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.buffer = deque(maxlen=10000)
        self.lock = threading.Lock()
        self._start_flusher()

    def log(self, inputs, prediction, **kwargs):
        with self.lock:
            self.buffer.append({
                'inputs': inputs,
                'prediction': prediction,
                **kwargs
            })
            if len(self.buffer) >= self.batch_size:
                self._flush()

    def _flush(self):
        if not self.buffer:
            return

        batch = []
        with self.lock:
            while self.buffer and len(batch) < self.batch_size:
                batch.append(self.buffer.popleft())

        if batch:
            self.monitor.log_batch(batch)

    def _start_flusher(self):
        def flush_periodically():
            while True:
                time.sleep(self.flush_interval)
                self._flush()

        thread = threading.Thread(target=flush_periodically, daemon=True)
        thread.start()

# Usage
batched_monitor = BatchedMonitor(monitor, batch_size=100)
```

### 2. Async Logging

Use async logging to prevent blocking:

```python
import asyncio
from whiteboxai import AsyncWhiteBoxAI

async def predict_and_log(model, X):
    # Make prediction (sync)
    prediction = model.predict(X)

    # Log asynchronously (non-blocking)
    await client.log_prediction_async(
        model_id=model_id,
        inputs=X,
        prediction=prediction
    )

    return prediction

# Initialize async client
client = AsyncWhiteBoxAI(api_key='your-api-key')

# Run
asyncio.run(predict_and_log(model, X))
```

### 3. Caching

Implement caching for repeated predictions:

```python
from functools import lru_cache
import hashlib
import json

class CachedMonitor:
    def __init__(self, monitor, cache_size=1000):
        self.monitor = monitor
        self.cache_size = cache_size

    def _hash_input(self, inputs):
        return hashlib.md5(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()

    @lru_cache(maxsize=1000)
    def predict(self, inputs_hash):
        # This would be called with actual prediction logic
        pass

    def predict_and_log(self, inputs):
        inputs_hash = self._hash_input(inputs)

        # Check cache
        cached_prediction = self.predict.cache_info()

        # Make prediction
        prediction = self.monitor.predict(inputs)

        # Log only if not cached
        if inputs_hash not in self.predict.cache_info():
            self.monitor.log_prediction(inputs, prediction)

        return prediction
```

### 4. Connection Pooling

Use connection pooling for database connections:

```python
from whiteboxai import WhiteBoxAI
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Configure connection pool
session = requests.Session()
retry = Retry(
    total=3,
    backoff_factor=0.3,
    status_forcelist=[500, 502, 503, 504]
)
adapter = HTTPAdapter(
    pool_connections=10,
    pool_maxsize=100,
    max_retries=retry
)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Use custom session
client = WhiteBoxAI(
    api_key='your-api-key',
    session=session
)
```

---

## Security Best Practices

### 1. API Key Management

```python
# DON'T: Hardcode API keys
client = WhiteBoxAI(api_key='sk-abc123...')  # BAD!

# DO: Use environment variables
client = WhiteBoxAI()  # Reads from WHITEBOXAI_API_KEY

# DO: Use secrets management
from your_secrets import get_secret
client = WhiteBoxAI(api_key=get_secret('whiteboxai-api-key'))
```

### 2. PII Detection and Masking

```python
from whiteboxai.security import PIIDetector, DataMasker

# Enable PII detection
pii_detector = PIIDetector()
masker = DataMasker()

def safe_log_prediction(inputs, prediction):
    # Detect PII
    pii_fields = pii_detector.detect(inputs)

    # Mask sensitive data
    if pii_fields:
        masked_inputs = masker.mask(inputs, pii_fields)
    else:
        masked_inputs = inputs

    # Log masked data
    client.log_prediction(
        model_id=model_id,
        inputs=masked_inputs,
        prediction=prediction,
        metadata={'pii_detected': bool(pii_fields)}
    )
```

### 3. TLS/SSL Configuration

```python
# Verify SSL certificates in production
client = WhiteBoxAI(
    api_key='your-api-key',
    verify_ssl=True,
    cert='/path/to/cert.pem'  # Optional custom cert
)
```

### 4. Rate Limiting

```python
from whiteboxai.utils import RateLimiter

# Implement client-side rate limiting
limiter = RateLimiter(
    requests_per_second=100,
    burst_size=200
)

@limiter.limit
def log_prediction(inputs, prediction):
    client.log_prediction(
        model_id=model_id,
        inputs=inputs,
        prediction=prediction
    )
```

---

## High Availability

### 1. Circuit Breaker Pattern

```python
from whiteboxai.resilience import CircuitBreaker

circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    timeout=60,
    expected_exception=Exception
)

@circuit_breaker
def resilient_predict_and_log(inputs):
    prediction = model.predict(inputs)
    client.log_prediction(
        model_id=model_id,
        inputs=inputs,
        prediction=prediction
    )
    return prediction

# Graceful degradation
try:
    prediction = resilient_predict_and_log(inputs)
except CircuitBreaker.CircuitBreakerOpen:
    # Circuit is open, skip logging
    prediction = model.predict(inputs)
    logger.warning("WhiteBoxAI circuit breaker open, prediction not logged")
```

### 2. Fallback Strategies

```python
class FallbackMonitor:
    def __init__(self, primary_client, fallback_logger=None):
        self.primary = primary_client
        self.fallback = fallback_logger or LocalLogger()

    def log_prediction(self, **kwargs):
        try:
            return self.primary.log_prediction(**kwargs)
        except Exception as e:
            logger.error(f"Primary logging failed: {e}")
            # Fallback to local logging
            return self.fallback.log(**kwargs)

class LocalLogger:
    def log(self, **kwargs):
        # Write to local file or queue
        with open('predictions.jsonl', 'a') as f:
            f.write(json.dumps(kwargs) + '\\n')
```

### 3. Health Checks

```python
from whiteboxai.health import HealthCheck

health_check = HealthCheck(client)

# Check API health
if not health_check.is_healthy():
    logger.error("WhiteBoxAI API is unhealthy")
    # Switch to degraded mode

# Expose health endpoint for orchestrators
@app.get("/health")
def health():
    return {
        "status": "healthy" if health_check.is_healthy() else "degraded",
        "whiteboxai": health_check.check_api(),
        "model": health_check.check_model(model_id)
    }
```

---

## Monitoring the Monitor

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge
from whiteboxai.metrics import PrometheusExporter

# Define metrics
predictions_logged = Counter(
    'whiteboxai_predictions_logged_total',
    'Total predictions logged to WhiteBoxAI'
)

logging_latency = Histogram(
    'whiteboxai_logging_latency_seconds',
    'Latency of logging operations'
)

api_errors = Counter(
    'whiteboxai_api_errors_total',
    'Total API errors',
    ['error_type']
)

# Instrument monitoring
class InstrumentedMonitor:
    def __init__(self, client):
        self.client = client

    def log_prediction(self, **kwargs):
        with logging_latency.time():
            try:
                result = self.client.log_prediction(**kwargs)
                predictions_logged.inc()
                return result
            except Exception as e:
                api_errors.labels(error_type=type(e).__name__).inc()
                raise
```

### Application Logs

```python
import logging
import structlog

# Structured logging
logger = structlog.get_logger()

def log_prediction_with_context(inputs, prediction):
    try:
        result = client.log_prediction(
            model_id=model_id,
            inputs=inputs,
            prediction=prediction
        )
        logger.info(
            "prediction_logged",
            model_id=model_id,
            prediction_id=result['id'],
            latency_ms=result.get('latency_ms')
        )
    except Exception as e:
        logger.error(
            "prediction_logging_failed",
            model_id=model_id,
            error=str(e),
            error_type=type(e).__name__
        )
        raise
```

---

## Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup-whiteboxai.sh

# Backup configuration
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/whiteboxai/${DATE}"

# Backup database
pg_dump whiteboxai_production > "${BACKUP_DIR}/database.sql"

# Backup model metadata
python -c "
from whiteboxai import WhiteBoxAI
client = WhiteBoxAI()
models = client.list_models()
with open('${BACKUP_DIR}/models.json', 'w') as f:
    json.dump(models, f)
"

# Upload to S3
aws s3 sync ${BACKUP_DIR} s3://whiteboxai-backups/${DATE}
```

### Recovery Procedures

```python
# Restore from backup
def restore_model(backup_file):
    with open(backup_file) as f:
        model_config = json.load(f)

    # Re-register model
    client.register_model(**model_config)

    # Restore baseline data
    if 'baseline' in model_config:
        client.set_baseline(
            model_id=model_config['id'],
            baseline_data=model_config['baseline']['data']
        )
```

---

## Scaling Strategies

### Horizontal Scaling

```python
# Use multiple workers for high-volume logging
from concurrent.futures import ThreadPoolExecutor
import queue

class ScalableMonitor:
    def __init__(self, client, num_workers=10):
        self.client = client
        self.queue = queue.Queue(maxsize=10000)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self._start_workers()

    def _start_workers(self):
        for _ in range(self.executor._max_workers):
            self.executor.submit(self._worker)

    def _worker(self):
        while True:
            batch = []
            # Collect batch
            for _ in range(100):
                try:
                    item = self.queue.get(timeout=1)
                    batch.append(item)
                except queue.Empty:
                    break

            # Log batch
            if batch:
                self.client.log_batch(
                    model_id=batch[0]['model_id'],
                    predictions=batch
                )

    def log(self, **kwargs):
        self.queue.put(kwargs)
```

### Kubernetes Deployment

```yaml
# whiteboxai-monitor.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service-with-monitoring
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
    spec:
      containers:
      - name: ml-service
        image: your-ml-service:latest
        env:
        - name: WHITEBOXAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: whiteboxai-secrets
              key: api-key
        - name: WHITEBOXAI_BASE_URL
          value: "https://api.whiteboxai.internal"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

---

## Resources

- [WhiteBoxAI API Documentation](https://docs.whiteboxai.com/api)
- [SDK Reference](https://docs.whiteboxai.com/sdk)
- [Best Practices](https://docs.whiteboxai.com/best-practices)
- [Support](mailto:whiteboxai-support@kpmg.com)

---

*Last Updated: December 29, 2024*
