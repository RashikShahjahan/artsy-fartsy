# AI Drawing Application

An interactive web application that allows users to generate and manipulate drawings using AI, with features to save drawings and find similar artwork.

## Tech Stack

- Frontend: React + TypeScript + Vite
- Backend: Bun + Express
- Database: PostgreSQL with pgvector
- AI: Claude API for code generation
- Drawing: Cairo graphics library
- Styling: TailwindCSS + DaisyUI

## Prerequisites

- Node.js ≥18.0.0
- Bun ≥1.0.0
- Python 3.11.x
- PostgreSQL with pgvector extension

## Local Development

1. **Clone and Install Dependencies**
   ```bash
   # Install frontend dependencies
   bun install
   
   # Install backend dependencies
   cd backend
   bun install
   
   # Create Python virtual environment
   python3 -m venv venv
   source venv/bin/activate
   pip install pycairo
   ```

2. **Environment Setup**
   Create a `.env` file in the root directory:
   ```
   ANTHROPIC_API_KEY=your_claude_api_key
   VOYAGE_API_KEY=your_voyage_api_key
   DATABASE_URL=postgres://postgres:postgres@localhost:5432/codeart
   ```

3. **Database Setup**
   ```bash
   # Start PostgreSQL with pgvector
   docker-compose -f backend/docker-compose.yml up -d
   ```

4. **Run Development Servers**
   ```bash
   # Terminal 1: Frontend
   bun run dev
   
   # Terminal 2: Backend
   cd backend
   bun run index.ts
   ```

## Deployment Options

### Heroku Deployment

1. **Prerequisites**
   - Heroku CLI installed
   - Heroku account with PostgreSQL addon

2. **Setup**
   ```bash
   heroku create your-app-name
   heroku addons:create heroku-postgresql
   heroku buildpacks:add heroku/python
   heroku buildpacks:add heroku/nodejs
   ```

3. **Configure Environment**
   ```bash
   heroku config:set ANTHROPIC_API_KEY=your_key
   heroku config:set VOYAGE_API_KEY=your_key
   heroku config:set NODE_ENV=production
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

### Docker Deployment

1. **Build Image**
   ```bash
   docker build -t ai-drawing-app .
   ```

2. **Run Container**
   ```bash
   docker run -d \
     -p 8000:8000 \
     -e ANTHROPIC_API_KEY=your_key \
     -e VOYAGE_API_KEY=your_key \
     -e DATABASE_URL=your_db_url \
     ai-drawing-app
   ```

