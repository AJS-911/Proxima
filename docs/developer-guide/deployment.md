# Deployment Guide for Proxima Agent

This guide covers deployment considerations and configuration for running Proxima Agent in production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Options](#deployment-options)
3. [Configuration](#configuration)
4. [Security Considerations](#security-considerations)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Scaling](#scaling)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
| --------- | ------- | ----------- |
| CPU       | 2 cores | 4+ cores    |
| RAM       | 4 GB    | 8+ GB       |
| Disk      | 10 GB   | 50+ GB      |
| Python    | 3.11    | 3.12        |

### Required Software

- Python 3.11 or later
- pip or uv package manager
- Docker (for containerized deployments)
- At least one quantum backend (Cirq, Qiskit, or LRET)

---

## Deployment Options

### Option 1: Direct Installation (Recommended for Development)

```bash
# Install from PyPI
pip install proxima-agent

# Or with all optional dependencies
pip install proxima-agent[all]

# Verify installation
proxima --version
```

### Option 2: Docker Container (Recommended for Production)

```bash
# Pull the official image
docker pull ghcr.io/yourusername/proxima:latest

# Run with default configuration
docker run -it ghcr.io/yourusername/proxima:latest

# Run with custom configuration
docker run -it \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/results:/app/results \
  ghcr.io/yourusername/proxima:latest \
  --config /app/configs/production.yaml run simulation.yaml
```

### Option 3: Docker Compose (Multi-Service)

```yaml
# docker-compose.prod.yml
version: "3.8"

services:
  proxima:
    image: ghcr.io/yourusername/proxima:latest
    environment:
      - PROXIMA_LOG_LEVEL=info
      - PROXIMA_LLM_API_KEY=${LLM_API_KEY}
    volumes:
      - ./configs:/app/configs:ro
      - ./results:/app/results
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          cpus: "4"
          memory: 8G
        reservations:
          cpus: "2"
          memory: 4G
    healthcheck:
      test: ["CMD", "proxima", "--version"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
# Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

### Option 4: Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: proxima-agent
  labels:
    app: proxima
spec:
  replicas: 1
  selector:
    matchLabels:
      app: proxima
  template:
    metadata:
      labels:
        app: proxima
    spec:
      containers:
        - name: proxima
          image: ghcr.io/yourusername/proxima:latest
          resources:
            limits:
              cpu: "4"
              memory: "8Gi"
            requests:
              cpu: "2"
              memory: "4Gi"
          env:
            - name: PROXIMA_LOG_LEVEL
              value: "info"
            - name: PROXIMA_LLM_API_KEY
              valueFrom:
                secretKeyRef:
                  name: proxima-secrets
                  key: llm-api-key
          volumeMounts:
            - name: config
              mountPath: /app/configs
            - name: results
              mountPath: /app/results
      volumes:
        - name: config
          configMap:
            name: proxima-config
        - name: results
          persistentVolumeClaim:
            claimName: proxima-results-pvc
---
apiVersion: v1
kind: Secret
metadata:
  name: proxima-secrets
type: Opaque
stringData:
  llm-api-key: "your-api-key-here"
```

---

## Configuration

### Environment Variables

| Variable              | Description              | Default                |
| --------------------- | ------------------------ | ---------------------- |
| `PROXIMA_LOG_LEVEL`   | Logging verbosity        | `info`                 |
| `PROXIMA_LLM_API_KEY` | LLM provider API key     | -                      |
| `PROXIMA_OUTPUT_DIR`  | Results output directory | `./results`            |
| `PROXIMA_CONFIG`      | Path to config file      | `configs/default.yaml` |
| `PROXIMA_BACKEND`     | Default quantum backend  | `auto`                 |

### Configuration Files

1. **Default Configuration** (`configs/default.yaml`)

   - Suitable for development and testing
   - Verbose logging enabled
   - All consent required

2. **Production Configuration** (`configs/production.yaml`)
   - Optimized for stability and security
   - Structured JSON logging
   - Rate limiting enabled

### Production Configuration Checklist

```yaml
# Key production settings
general:
  verbosity: info # Not debug
  color_enabled: false # For log aggregation

llm:
  require_consent: true # Always require consent

consent:
  auto_approve_remote_llm: false # Never auto-approve remote
  remember_decisions: false # Don't persist decisions

logging:
  structured: true # JSON logging
  mask_sensitive_data: true # Security

security:
  validate_inputs: true # Input validation
```

---

## Security Considerations

### API Key Management

```bash
# Never commit API keys to version control
# Use environment variables or secrets management

# Docker secrets
docker secret create llm_api_key /path/to/key.txt

# Kubernetes secrets
kubectl create secret generic proxima-secrets \
  --from-literal=llm-api-key=sk-xxx
```

### Network Security

1. **Disable Remote LLM** if not needed:

   ```yaml
   llm:
     provider: none
   ```

2. **Use Local LLM** for sensitive data:

   ```yaml
   llm:
     provider: ollama
     local_endpoint: "http://localhost:11434"
   ```

3. **Restrict network access** in containers:
   ```yaml
   # docker-compose.yml
   services:
     proxima:
       network_mode: "none" # No network access
   ```

### File System Security

```yaml
# Production security settings
security:
  max_file_size_mb: 100
  allowed_extensions:
    - .yaml
    - .yml
    - .json
```

---

## Monitoring and Logging

### Structured Logging

```python
# Log output format (JSON)
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "info",
  "event": "execution_complete",
  "trace_id": "abc-123",
  "backend": "cirq",
  "duration_ms": 45.2
}
```

### Health Checks

```bash
# Docker health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD proxima --version || exit 1

# Kubernetes liveness probe
livenessProbe:
  exec:
    command: ["proxima", "--version"]
  initialDelaySeconds: 10
  periodSeconds: 30
```

### Metrics Collection

```yaml
# Prometheus metrics (if enabled)
metrics:
  enabled: true
  port: 9090
  path: /metrics
```

### Log Aggregation

For production, integrate with log aggregation systems:

- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **Loki + Grafana**: Lightweight log aggregation
- **Cloud Services**: AWS CloudWatch, GCP Logging, Azure Monitor

---

## Scaling

### Horizontal Scaling

Proxima Agent is stateless and can be scaled horizontally:

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: proxima-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: proxima-agent
  minReplicas: 1
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### Resource Limits

```yaml
# Docker Compose
services:
  proxima:
    deploy:
      resources:
        limits:
          cpus: "4"
          memory: 8G
        reservations:
          cpus: "2"
          memory: 4G
```

### Parallel Backend Execution

```yaml
# Enable parallel execution in production
backends:
  parallel_execution: true
  max_workers: 4
```

---

## Troubleshooting

### Common Issues

1. **Memory Errors**

   ```bash
   # Increase memory limit
   docker run -m 8g ghcr.io/yourusername/proxima:latest
   ```

2. **Backend Not Found**

   ```bash
   # Install specific backend
   pip install proxima-agent[cirq]  # or [qiskit]
   ```

3. **LLM Connection Failed**
   ```yaml
   # Check configuration
   llm:
     local_endpoint: "http://localhost:11434"
     request_timeout_seconds: 60
   ```

### Debug Mode

```bash
# Enable debug logging
PROXIMA_LOG_LEVEL=debug proxima run simulation.yaml

# Or in config
general:
  verbosity: debug
```

### Health Check Commands

```bash
# Check installation
proxima --version

# List available backends
proxima backends list

# Validate configuration
proxima config validate --config production.yaml

# Test LLM connection
proxima llm test
```

---

## Backup and Recovery

### Data Backup

```bash
# Backup results and configuration
tar -czvf proxima-backup-$(date +%Y%m%d).tar.gz \
  ./configs \
  ./results \
  ./logs
```

### Disaster Recovery

1. Store configuration in version control
2. Use persistent volumes for results
3. Implement log retention policies
4. Regular backup of simulation results

---

## Support

For production support:

1. Check [documentation](https://proxima.readthedocs.io)
2. Review [GitHub Issues](https://github.com/yourusername/proxima/issues)
3. Contact maintainers for enterprise support

---

_Last updated: January 2024_
