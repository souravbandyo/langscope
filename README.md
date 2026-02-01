# LangScope

**Multi-domain LLM Evaluation Framework using TrueSkill + Plackett-Luce**

LangScope is a comprehensive framework for evaluating large language models across multiple domains including text, code, speech, and vision. It combines mathematical rigor with practical evaluation needs.

## Project Structure

```
langscope/
├── Algorithm/          # Backend API (Python/FastAPI)
│   ├── langscope/      # Core library
│   ├── test/           # Test suite
│   └── requirements.txt
│
└── frontend/           # Frontend UI (React/TypeScript)
    ├── src/
    │   ├── api/        # API hooks and client
    │   ├── components/ # Reusable UI components
    │   ├── pages/      # Page components
    │   └── store/      # State management
    └── package.json
```

## Features

| Feature | Description |
|---------|-------------|
| **10-Dimensional Ratings** | Compare models on quality, cost, speed, reliability |
| **Domain-Specific Rankings** | Medical, legal, coding, multilingual leaderboards |
| **Transfer Learning** | Get ratings for new domains via similar evaluated domains |
| **Ground Truth Evaluation** | Objective metrics for ASR, TTS, code execution, visual QA |
| **Cost-Adjusted Scores** | Find the best model within your budget |
| **Specialist Detection** | Identify models that excel in specific niches |

## Getting Started

### Backend Setup

```bash
cd Algorithm
pip install -r requirements.txt
cp env.template .env
# Edit .env with your configuration
uvicorn langscope.api.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env
# Edit .env with your configuration
npm run dev
```

## API Documentation

Once running, access the API docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Tech Stack

**Backend:**
- Python 3.9+
- FastAPI
- MongoDB
- Redis (optional - caching)
- Qdrant (optional - semantic search)

**Frontend:**
- React 18
- TypeScript
- React Query
- Zustand
- Vite

## License

See LICENSE file for details.
