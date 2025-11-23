# Deployment Guide for coTherapist

This guide covers different deployment scenarios for coTherapist.

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Cloud Deployment](#cloud-deployment)
3. [API Server](#api-server)
4. [Docker Deployment](#docker-deployment)
5. [Production Considerations](#production-considerations)

---

## Local Deployment

### For Development/Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive chat
python scripts/chat.py

# Or use Python API
python -c "
from coTherapist.pipeline import CoTherapistPipeline
pipeline = CoTherapistPipeline()
pipeline.setup()
result = pipeline.generate_response('I feel anxious')
print(result['response'])
"
```

### Hardware Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB
- GPU: Optional (CPU inference supported)

**Recommended**:
- CPU: 8+ cores
- RAM: 16GB
- Storage: 20GB
- GPU: NVIDIA with 8GB+ VRAM (for faster inference)

---

## Cloud Deployment

### AWS EC2

**Instance Recommendations**:
- `g4dn.xlarge` (with GPU) - Best performance
- `m5.xlarge` (CPU only) - Budget option

**Setup**:
```bash
# Launch EC2 instance
# SSH into instance
ssh -i key.pem ubuntu@<instance-ip>

# Install dependencies
sudo apt update
sudo apt install python3-pip git
git clone https://github.com/ai4mhx/coTherapist.git
cd coTherapist
pip3 install -r requirements.txt

# Run application
python3 scripts/chat.py
```

### Google Cloud Platform

**Compute Engine**:
```bash
# Create VM with GPU
gcloud compute instances create cotherapist-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1

# SSH and setup
gcloud compute ssh cotherapist-vm
# ... same setup as AWS
```

### Azure

**Virtual Machine**:
```bash
# Create VM
az vm create \
  --resource-group coTherapist \
  --name cotherapist-vm \
  --image UbuntuLTS \
  --size Standard_NC6

# Setup similar to above
```

---

## API Server

### FastAPI Implementation

Create `api_server.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from coTherapist.pipeline import CoTherapistPipeline
import uvicorn

app = FastAPI(title="coTherapist API")

# Initialize pipeline
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = CoTherapistPipeline()
    pipeline.setup()

class ChatRequest(BaseModel):
    message: str
    return_details: bool = False

class ChatResponse(BaseModel):
    response: str
    safe: bool
    crisis_detected: bool
    details: dict = None

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    result = pipeline.generate_response(
        request.message,
        return_details=request.return_details
    )
    
    return ChatResponse(**result)

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": pipeline is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run Server**:
```bash
pip install fastapi uvicorn
python api_server.py
```

**Test API**:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "I feel anxious"}'
```

---

## Docker Deployment

### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install coTherapist
RUN pip install -e .

# Expose port for API (if using)
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "scripts/chat.py"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  cotherapist:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - LOG_LEVEL=INFO
    command: python api_server.py
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Build and Run**:
```bash
# Build
docker build -t cotherapist .

# Run interactive
docker run -it cotherapist

# Run API server
docker-compose up -d
```

---

## Production Considerations

### Security

1. **API Authentication**
   ```python
   from fastapi.security import HTTPBearer
   
   security = HTTPBearer()
   
   @app.post("/chat")
   async def chat(request: ChatRequest, token: str = Depends(security)):
       # Verify token
       pass
   ```

2. **Rate Limiting**
   ```python
   from slowapi import Limiter
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   
   @app.post("/chat")
   @limiter.limit("10/minute")
   async def chat(request: ChatRequest):
       pass
   ```

3. **HTTPS/SSL**
   - Use Let's Encrypt for certificates
   - Configure reverse proxy (nginx)
   - Enable SSL in uvicorn

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cotherapist.log'),
        logging.StreamHandler()
    ]
)
```

### Monitoring

1. **Health Checks**
   ```python
   @app.get("/health")
   async def health_check():
       return {
           "status": "healthy",
           "model_loaded": pipeline is not None,
           "timestamp": datetime.now().isoformat()
       }
   ```

2. **Metrics**
   ```python
   from prometheus_client import Counter, Histogram
   
   request_counter = Counter('requests_total', 'Total requests')
   response_time = Histogram('response_time_seconds', 'Response time')
   ```

3. **Error Tracking**
   - Use Sentry for error tracking
   - Log all exceptions
   - Monitor crisis detections

### Scaling

1. **Horizontal Scaling**
   ```yaml
   # Kubernetes deployment
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: cotherapist
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: cotherapist
     template:
       spec:
         containers:
         - name: cotherapist
           image: cotherapist:latest
   ```

2. **Load Balancing**
   - Use nginx or HAProxy
   - Distribute requests across instances
   - Health check endpoints

3. **Caching**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def get_cached_embedding(text):
       return embedding_model.encode(text)
   ```

### Database (Optional)

For conversation logging:

```python
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(String, primary_key=True)
    user_message = Column(String)
    bot_response = Column(String)
    timestamp = Column(DateTime)
    crisis_detected = Column(Boolean)
```

### Backup and Recovery

1. **Model Backups**
   ```bash
   # Automated backup
   crontab -e
   0 2 * * * rsync -av /app/models /backup/models
   ```

2. **Configuration Backups**
   ```bash
   # Version control
   git add config/
   git commit -m "Update config"
   git push
   ```

### Legal and Compliance

1. **Terms of Service**: Clear usage terms
2. **Privacy Policy**: Data handling disclosure
3. **Disclaimers**: Not a replacement for therapy
4. **HIPAA Compliance**: If handling PHI (US)
5. **GDPR Compliance**: If serving EU users
6. **Audit Logs**: Track all interactions

### Crisis Response

1. **Automated Alerts**
   ```python
   if result['crisis_detected']:
       send_alert_to_monitoring_team(user_id, message)
       log_crisis_event(details)
   ```

2. **Emergency Contacts**
   - Display prominently
   - Update regularly
   - Multi-language support

### Testing in Production

1. **A/B Testing**: Compare model versions
2. **Canary Deployments**: Gradual rollout
3. **Shadow Mode**: Test alongside human therapists

### Performance Optimization

1. **Model Optimization**
   - Use quantized models (4-bit/8-bit)
   - Enable Flash Attention
   - Batch inference when possible

2. **Caching**
   - Cache embeddings
   - Cache frequent responses
   - Use Redis for distributed cache

3. **Async Processing**
   ```python
   @app.post("/chat")
   async def chat(request: ChatRequest):
       result = await asyncio.to_thread(
           pipeline.generate_response,
           request.message
       )
       return result
   ```

---

## Maintenance

### Regular Updates

- **Model Retraining**: Monthly with new data
- **Dependency Updates**: Weekly security patches
- **Knowledge Base**: Quarterly review
- **Crisis Keywords**: Monthly review

### Monitoring Checklist

- [ ] Server uptime
- [ ] Response times
- [ ] Error rates
- [ ] Crisis detection rate
- [ ] User satisfaction
- [ ] Model performance metrics

---

## Support

For deployment issues:
- GitHub Issues: [Report Issue](https://github.com/ai4mhx/coTherapist/issues)
- Documentation: See README.md and ARCHITECTURE.md
- Community: GitHub Discussions

⚠️ **Important**: Always deploy with proper safety mechanisms and human oversight for mental healthcare applications.
