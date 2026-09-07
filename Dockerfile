# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install system dependencies (git for repo cloning, nodejs/npm for frontend)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Install Frontend dependencies
COPY code-graph/package*.json ./code-graph/
RUN cd code-graph && npm ci

# 3. Copy project source code
COPY . .

# 4. Build Next.js frontend
RUN cd code-graph && npm run build

# Expose Next.js frontend port
EXPOSE 3000

# Default command: Launch CodeGraph AI web visualization
CMD ["npm", "run", "start", "--prefix", "code-graph"]