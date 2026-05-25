#!/bin/bash
# Quick start script for BrainAI

set -e

echo "🧠 BrainAI — Quick Start"
echo "=========================="

# Check weights
if [ ! -f "backend/weights/unet_best.pth" ]; then
  echo "⚠️  WARNING: Model weights not found at backend/weights/unet_best.pth"
  echo "   The system will run in DEMO MODE (random weights)."
  echo "   Copy your trained model: cp your_model.pth backend/weights/unet_best.pth"
  echo ""
fi

# Check if running Docker or local
if command -v docker-compose &> /dev/null; then
  echo "🐳 Starting with Docker Compose..."
  docker-compose up --build
else
  echo "🖥️  Starting locally..."
  echo ""
  echo "--- Starting Backend ---"
  cd backend
  pip install -r requirements.txt -q
  uvicorn app.main:app --host 0.0.0.0 --port 8000 &
  BACKEND_PID=$!
  echo "✓ Backend running at http://localhost:8000"
  echo ""
  cd ..

  echo "--- Starting Frontend ---"
  cd frontend
  npm install --legacy-peer-deps -s
  npm start &
  FRONTEND_PID=$!
  cd ..

  echo ""
  echo "✅ BrainAI is running!"
  echo "   Frontend: http://localhost:3000"
  echo "   API Docs: http://localhost:8000/docs"
  echo ""
  echo "Press Ctrl+C to stop."

  trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" INT TERM
  wait
fi
