# Use multi-stage build
FROM oven/bun:latest as builder

# Install Python and required dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libcairo2-dev \
    pkg-config \
    python3-dev

WORKDIR /app

# Copy shared folder first
COPY shared/ /app/shared/
WORKDIR /app/shared
RUN bun install

# Copy and build frontend
COPY frontend/ /app/frontend/
WORKDIR /app/frontend
RUN bun install
RUN bun run build 

# Copy and setup backend
COPY backend/ /app/backend/
WORKDIR /app/backend
RUN bun install

# Create Python virtual environment and install dependencies
RUN python3 -m venv venv
RUN venv/bin/pip install pycairo

# Start production image
FROM oven/bun:latest

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    libcairo2

WORKDIR /app

# Copy built assets and backend
COPY --from=builder /app/frontend/dist ./dist
COPY --from=builder /app/backend ./backend
COPY --from=builder /app/backend/venv ./backend/venv
COPY --from=builder /app/shared ./shared

# Set environment variables
ENV NODE_ENV=production
ENV PORT=8000

# Expose port
EXPOSE 8000

# Start the application
CMD ["bun", "run", "backend/index.ts"]