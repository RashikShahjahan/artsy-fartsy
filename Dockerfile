# Use multi-stage build
FROM oven/bun:latest as builder

# Install Python and required dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    postgresql \
    postgresql-contrib \
    postgresql-server-dev-all \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pgvector extension
RUN git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install

WORKDIR /app

# Copy package files first for better caching
COPY shared/package*.json /app/shared/
COPY frontend/package*.json /app/frontend/
COPY backend/package*.json /app/backend/

# Install dependencies
WORKDIR /app/shared
RUN bun install

WORKDIR /app/frontend
RUN bun install

WORKDIR /app/backend
RUN bun install

# Copy rest of the application
COPY shared/ /app/shared/
COPY frontend/ /app/frontend/
COPY backend/ /app/backend/

# Build frontend
WORKDIR /app/frontend
RUN bun run build

# Setup Python environment
WORKDIR /app/backend/art_libraries
RUN python3 -m venv venv
RUN . venv/bin/activate && pip install pycairo && pip install musicpy

# Start production image
FROM oven/bun:latest

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    libcairo2 \
    postgresql \
    postgresql-contrib \
    postgresql-server-dev-all \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pgvector extension
RUN git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install

WORKDIR /app

# Copy built assets and backend
COPY --from=builder /app/frontend/dist ./frontend/dist
COPY --from=builder /app/backend ./backend
COPY --from=builder /app/backend/art_libraries/venv ./backend/art_libraries/venv
COPY --from=builder /app/shared ./shared

# Create a script to initialize the database
RUN echo '#!/bin/bash\n\
if [ -n "$DATABASE_URL" ]; then\n\
  echo "Initializing database with vector extension..."\n\
  # Parse DATABASE_URL\n\
  DB_URL=${DATABASE_URL}\n\
  # Extract username and password\n\
  DB_USER=$(echo $DB_URL | sed -E "s/^postgres:\\/\\/([^:]+):.*/\\1/")\n\
  DB_PASS=$(echo $DB_URL | sed -E "s/^postgres:\\/\\/[^:]+:([^@]+).*/\\1/")\n\
  # Extract host, port and dbname\n\
  DB_HOST=$(echo $DB_URL | sed -E "s/^postgres:\\/\\/[^@]+@([^:]+).*/\\1/")\n\
  DB_PORT=$(echo $DB_URL | sed -E "s/^postgres:\\/\\/[^@]+@[^:]+:([0-9]+).*/\\1/")\n\
  DB_NAME=$(echo $DB_URL | sed -E "s/^postgres:\\/\\/[^@]+@[^:]+:[0-9]+\\/([^?]+).*/\\1/")\n\
  echo "Connecting to PostgreSQL at $DB_HOST:$DB_PORT/$DB_NAME as $DB_USER"\n\
  PGPASSWORD=$DB_PASS psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f /app/backend/setup-vector-extension.sql || echo "Database initialization failed but continuing"\n\
  echo "Database initialization completed"\n\
fi\n\
cd backend && bun run src/index.ts\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set environment variables
ENV NODE_ENV=production
ENV PATH="/app/backend/art_libraries/venv/bin:$PATH"
ENV PORT=8000

# Expose port
EXPOSE $PORT

# Start the application
CMD ["/app/entrypoint.sh"]