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

# Copy package files
COPY package*.json ./
COPY backend/package*.json ./backend/

# Install dependencies
RUN bun install
RUN cd backend && bun install

# Copy source code
COPY . .

# Create Python virtual environment and install dependencies
RUN python3 -m venv backend/venv
RUN backend/venv/bin/pip install pycairo

# Build frontend
RUN bun run build

# Start production image
FROM oven/bun:latest

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    libcairo2

WORKDIR /app

# Copy built assets and backend
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/backend ./backend
COPY --from=builder /app/backend/venv ./backend/venv

# Set environment variables
ENV NODE_ENV=production
ENV PORT=8000

# Expose port
EXPOSE 8000

# Start the application
CMD ["bun", "run", "backend/index.ts"] 