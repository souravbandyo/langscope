# LangScope Algorithm

**Multi-domain LLM Evaluation Framework using TrueSkill + Plackett-Luce**

LangScope is a comprehensive framework for evaluating large language models across multiple domains including text, code, speech, and vision. It combines mathematical rigor with practical evaluation needs.

## Why LangScope?

**Problem:** How do you choose the right LLM for your specific use case?

Traditional approaches fall short:
- **Static benchmarks** (MMLU, HumanEval) don't reflect real-world performance in your domain
- **Single scores** hide critical trade-offs—a model may be great at quality but too slow or expensive
- **Generic leaderboards** don't account for specialized domains (Hindi medical, legal contracts, etc.)

**LangScope Solution:** Domain-specific, multi-dimensional rankings that tell you:
- Which model is **best for medical Q&A**? For **legal document review**? For **Hindi customer support**?
- Which model gives the **best value for money** in your domain?
- Which model has the **lowest latency** while maintaining quality?
- Which model **hallucinates least** in high-stakes domains?

### Key Features

| Feature | Benefit |
|---------|---------|
| **10-Dimensional Ratings** | Compare models on quality, cost, speed, reliability—not just one score |
| **Domain-Specific Rankings** | Medical, legal, coding, multilingual—each domain has its own leaderboard |
| **Transfer Learning** | Get ratings even for new domains by leveraging similar evaluated domains |
| **Ground Truth Evaluation** | Objective metrics for ASR, TTS, code execution, visual QA |
| **Cost-Adjusted Scores** | Find the best model within your budget |
| **Specialist Detection** | Identify models that excel in specific niches |

---

## Core Concepts

### TrueSkill + Plackett-Luce Rating System

Each model has a skill rating represented as a Gaussian distribution:
- **μ (mu)**: Mean skill estimate
- **σ (sigma)**: Uncertainty in the estimate

The Plackett-Luce model provides probability of ranking outcomes:
```
P(r₁ > r₂ > ... > rₙ) = ∏ᵢ (λᵣᵢ / Σⱼ≥ᵢ λᵣⱼ)
```

After each match, ratings update using Bayesian inference:
```
μ_new = μ + (σ²/c) × v(t)
σ_new = σ × √(1 - (σ²/c²) × w(t))
```

**Conservative estimate**: `μ - 3σ` (99.7% confidence lower bound)

### 10-Dimensional Evaluation

Models are rated across 10 dimensions, each with its own TrueSkill rating:

| Dimension | Formula | What It Measures |
|-----------|---------|------------------|
| **Raw Quality** | `μ_raw` from judge rankings | Pure response quality |
| **Cost-Adjusted** | `μ / log(1 + cost/M)` | Value for money |
| **Latency** | `1 / (1 + L/1000ms)` | Response speed |
| **TTFT** | `1 / (1 + T/200ms)` | Time to first token (streaming) |
| **Consistency** | `1 / (1 + σ_responses)` | Output stability (n=5 runs) |
| **Token Efficiency** | `μ / log(1 + tokens)` | Quality per output token |
| **Instruction Following** | `satisfied / total` | Format/constraint compliance |
| **Hallucination Resistance** | `1 - (halluc / claims)` | Factual accuracy |
| **Long Context** | `quality@max / quality@4K` | Performance at long contexts |
| **Combined** | `Σ wᵢ × μᵢ` | Weighted aggregate (customizable) |

**Default Weights:**
- Raw Quality: 20%, Instruction Following: 15%, Hallucination Resistance: 15%
- Cost-Adjusted: 10%, Latency: 10%, Consistency: 10%, Token Efficiency: 10%
- TTFT: 5%, Long Context: 5%

---

## Evaluation Modes

### 1. Subjective Evaluation (Peer-Federated)

For domains where quality is subjective (creative writing, general QA):

1. **Swiss Pairing**: Group models with similar ratings (Δμ < 75)
2. **Role Assignment**: Based on strata (Elite μ≥1520, High 1450-1519, Mid 1400-1449, Low <1400)
   - Elite models: Can be judges and content creators
   - Lower strata: Primarily competitors
3. **Match Execution**: 
   - Content creator generates a case/question
   - All players respond
   - Judge models rank the responses
4. **Rating Update**: Weighted Borda aggregation → TrueSkill update

### 2. Ground Truth Evaluation (Objective Metrics)

For domains with measurable correctness:

| Domain | Primary Metric | How It Works |
|--------|---------------|--------------|
| **ASR** (Speech Recognition) | WER (Word Error Rate) | Compare transcription to reference text |
| **TTS** (Text-to-Speech) | Composite Score | Round-trip WER + UTMOS quality + SNR |
| **Code Execution** | pass@k | Run generated code against test cases |
| **Visual QA** | Accuracy | Compare answers to ground truth |
| **Needle in Haystack** | Retrieval Accuracy | Find specific info in long contexts |
| **Long Document QA** | Answer Accuracy | Answer questions about documents |

**Ground Truth Metrics Available:**

```python
# Speech Recognition
WER, CER, MER, WIL  # Word/Character/Match Error Rate, Word Information Lost

# Text-to-Speech  
round_trip_wer      # TTS → ASR → compare to original
utmos               # Neural MOS predictor (1-5 scale)
snr                 # Signal-to-noise ratio
speaker_similarity  # Voice consistency

# Code Execution
syntax_valid        # Code parses correctly
tests_pass          # Passes test cases (pass@k)
execution_time      # Runtime performance

# Document Understanding
answer_accuracy     # Exact/fuzzy match
citation_precision  # Correct source attribution
retrieval_accuracy  # Found the needle
```

---

## Automatic Domain Classification

LangScope automatically routes prompts to the appropriate evaluation workflow using a hierarchical classifier.

### Classification Hierarchy

**Stage 1: Category Detection**
```
Input: "What is the treatment for diabetes?"
→ Category: core_language (vs multimodal, safety, technical, etc.)
```

**Stage 2: Domain Detection**
```
→ Base Domain: medical
→ Template: clinical_reasoning
```

**Stage 3: Language Detection**
```
Input: "मधुमेह का इलाज क्या है?"
→ Variant: hindi
→ Full Domain: hindi_medical
```

### Supported Categories and Domains

| Category | Domains | Evaluation Type |
|----------|---------|-----------------|
| **Core Language** | medical, legal, financial, customer_support, education, code_generation | Subjective |
| **Multimodal** | asr, tts, visual_qa, document_extraction, image_captioning, ocr, video | Ground Truth |
| **Safety** | bias_detection, harmful_content, privacy, truthfulness | Subjective |
| **Cultural** | cultural_competence, regional_language, religious, local_context | Subjective |
| **Technical** | scientific, mathematical, logical, data_analysis | Mixed |
| **Long Context** | needle_in_haystack, long_document_qa, multi_document, code_completion | Ground Truth |
| **Creative** | creative_writing, tone_adaptation | Subjective |

### Multilingual Support

Automatic language detection for:
- **Hindi** (Devanagari script)
- **Bengali** (Bengali script)
- **Tamil** (Tamil script)
- **Telugu** (Telugu script)
- **Odia** (Odia script)
- **Marathi** (Devanagari + markers)

Example:
```python
classify("मधुमेह का इलाज क्या है?")
# → category: "core_language", base_domain: "medical", variant: "hindi"
# → template_name: "hindi_medical"
```

---

## Sample Datasets

Ground truth evaluation uses stratified sample datasets:

| Domain | Samples | Difficulty Levels | Description |
|--------|---------|-------------------|-------------|
| `needle_in_haystack` | 8 | easy, medium, hard | Context lengths 4K-128K tokens |
| `asr` | 8 | easy-hard | Clean, noisy, accented, technical speech |
| `visual_qa` | 10 | easy-hard | Counting, color, object, scene, reasoning |
| `code_completion` | 8 | easy-hard | HumanEval/MBPP style problems |
| `long_document_qa` | 5 | medium-hard | Legal, research, technical, financial, medical |

**Sample Format:**
```json
{
  "id": "needle_001",
  "difficulty": "medium",
  "context_length": 32000,
  "needle": "The secret code is ALPHA-7749",
  "haystack": "... long document ...",
  "question": "What is the secret code?",
  "expected_answer": "ALPHA-7749"
}
```

---

## Multi-Layer Caching

LangScope uses a 4-layer caching architecture for performance:

```
Request → LOCAL (in-memory) → REDIS (distributed) → QDRANT (vector) → MONGODB (persistent)
```

### Cache Categories

| Category | Layers | TTL | Purpose |
|----------|--------|-----|---------|
| **Leaderboard** | Local, Redis | 5 min | Pre-computed domain rankings |
| **Model Ratings** | Local, Redis | 10 min | Individual model ratings by domain |
| **Domain Classification** | Local, Redis, MongoDB | 24 hours | Prompt → domain mapping results |
| **Transfer Predictions** | Qdrant, MongoDB | 7 days | Cross-domain rating predictions |
| **Session** | Local, Redis, MongoDB | 30 min | Arena session state |
| **Rate Limit** | Redis | 1 min | API rate limiting |

### Semantic Cache for Evaluation Results

LangScope caches evaluation results so similar evaluation queries can be served faster:

```bash
# First evaluation request - runs full evaluation
GET /leaderboard/medical?dimension=raw_quality

# Similar request with slight variation - cache hit
GET /leaderboard/clinical?dimension=raw_quality
# → Domain similarity: 0.94 > threshold
# → Return cached leaderboard (domains are highly correlated)
```

**What gets cached:**
- Leaderboard computations by domain/dimension
- Model rating lookups
- Domain classification results
- Transfer learning predictions

**Cache invalidation:**
- Automatic on new match results
- Manual via `/cache/invalidate/{category}`

---

## Transfer Learning (Model Rank API)

LangScope's transfer learning system enables ratings for **any model in any domain across all 10 dimensions**—even when direct observations don't exist. This powers the Model Rank API.

### How It Works

```
GET /models/{id}/rating?domain=X&dimension=Y
  │
  ├── Direct rating exists? → Return direct rating (source="direct")
  │
  └── No direct rating? → Transfer Learning computes prediction (source="transfer")
      ├── Find similar domains (faceted similarity)
      ├── Multi-source weighted transfer
      └── Return predicted μ, σ with confidence score
```

### Faceted Domain Similarity

Domains are decomposed into 5 facets for nuanced similarity:

| Facet | Examples | Weight |
|-------|----------|--------|
| **Field** | medical, legal, financial, code | 35% |
| **Language** | english, hindi, bengali, chinese | 20% |
| **Modality** | text, code, audio, image | 20% |
| **Task** | qa, summarization, generation, detection | 15% |
| **Specialty** | radiology, contracts, derivatives | 10% |

**Composite similarity:**
```
ρ(source, target) = Σ βₖ × sim_k(source_k, target_k)
```

### Bayesian Similarity Learning

Facet similarities blend expert priors with observed data:

```
similarity = α × prior + (1 - α) × observed
α = 1 / (1 + n/τ)  # α decreases as observations grow
```

**Example priors:**
- Bengali ↔ Odia: 0.75 (same script family)
- Hindi ↔ Urdu: 0.85 (mutual intelligibility)
- Medical ↔ Clinical: 0.90 (field overlap)

### Multi-Source Transfer

When predicting ratings, LangScope combines signals from up to 7 similar domains:

```
μ_target = Σ wᵢ × μ_source_i   (reliability-weighted)
σ_target = √(σ²_pooled + uncertainty_penalty)

Reliability: rᵢ = ρᵢ² / σ_source_i²
```

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /transfer/models/{id}/rating` | Get rating (direct or transferred) |
| `GET /transfer/models/{id}/ratings` | All 10 dimensions |
| `GET /transfer/domains/{domain}/similar` | Find similar domains |
| `GET /transfer/domains/similarity` | Explain domain correlation |
| `GET /transfer/leaderboard/{domain}` | Transfer-aware leaderboard |

### Specialist Detection

Models are classified based on performance variance across domains:

| Type | Description | Example |
|------|-------------|---------|
| **Specialist** | High variance, excels in specific domains | Code-focused model |
| **Generalist** | Low variance, consistent across domains | GPT-4 |
| **Mixed** | Moderate variance | Most models |

---

## User Feedback (Arena Mode)

Users can provide direct feedback through battle sessions:

### How It Works

1. **Start Session**: User begins an arena session for a use-case
2. **Present Battles**: Show responses from 2-4 models
3. **Collect Rankings**: User ranks the responses
4. **Compute Deltas**: Compare user feedback to predictions
5. **Update Ratings**: Apply user feedback with 2× weight

### Zero-Sum Conservation

Rating changes are conserved:
```
Σᵢ Δᵢ = 0  (total rating points unchanged)
```

### Use-Case Adjustments

Different use-cases may prefer different models:
```python
# User "alice" doing "patient_education" prefers model X
# User "bob" doing "research_summary" prefers model Y

recommendations = get_recommendations(
    use_case="patient_education",
    user_id="alice"
)
# Returns personalized model rankings
```

---

## Installation

### Requirements

- **Python 3.9+**
- **MongoDB** (required)
- **Redis** (optional - for caching and sessions)
- **Qdrant** (optional - for semantic cache)

### Setup

```bash
# Clone repository
git clone https://github.com/souravbandyo/langscope-algorithm.git
cd langscope-algorithm

# Install dependencies
pip install -r requirements.txt

# Optional: Install classification model
pip install sentence-transformers

# Configure environment
cp env.template .env
# Edit .env with your settings
```

### Environment Variables

```bash
# Required
MONGODB_URI=mongodb://localhost:27017
DB_NAME=langscope

# Optional - Caching
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333

# Authentication (Supabase)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_JWT_SECRET=your-jwt-secret

# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
XAI_API_KEY=xai-...
HUGGINGFACE_API_KEY=hf_...
```

---

## LLM Providers

LangScope includes a unified LLM provider module for integrating with multiple LLM services:

| Provider | Models | Features |
|----------|--------|----------|
| **OpenAI** | GPT-4o, GPT-4, GPT-3.5 | Sync/async, streaming |
| **Anthropic** | Claude 3.5, Claude 3 | Sync/async, streaming |
| **Groq** | Llama 3, Mixtral | Ultra-fast inference |
| **XAI** | Grok-2, Grok-3 | Real-time data |
| **HuggingFace** | Gemma, Llama, Qwen, Mistral | Inference API + local models |

### Usage

```python
from langscope.llm import LLMFactory, Message, LLMConfig

# Get a provider
provider = LLMFactory.get_provider("openai")

# Generate a response
response = await provider.generate([
    Message(role="user", content="Explain neural networks")
])
print(response.content)

# With custom config
config = LLMConfig(model="gpt-4o", temperature=0.7, max_tokens=1000)
response = await provider.generate(messages, config=config)

# Streaming
async for chunk in provider.generate_stream(messages):
    print(chunk, end="")

# List available providers
from langscope.llm import list_providers
print(list_providers())  # ['openai', 'anthropic', 'groq', 'xai', 'huggingface']
```

### HuggingFace Local Models

```python
from langscope.llm.providers.huggingface import HuggingFaceProvider

# Use local transformers instead of Inference API
provider = HuggingFaceProvider(use_local=True, device="cuda")
response = provider.generate_sync([
    Message(role="user", content="Hello!")
])
```

---

## Authentication

LangScope uses **Supabase** for stateless JWT authentication.

### Authentication Methods

| Method | Header | Use Case |
|--------|--------|----------|
| **JWT Bearer** | `Authorization: Bearer <token>` | User authentication |
| **Service Role** | `X-Service-Role-Key: <key>` | Server-to-server |
| **API Key** | `X-API-Key: <key>` | Legacy support |

### Protected Endpoints

All API endpoints (except `/health` and `/docs`) require authentication.

```bash
# User request with JWT
curl -H "Authorization: Bearer eyJ..." http://localhost:8000/leaderboard/medical

# Service-to-service with service role key
curl -H "X-Service-Role-Key: your-service-key" http://localhost:8000/models
```

### User Context

Authenticated requests include user context:

```python
from langscope.api.middleware.auth import get_current_user

@router.get("/profile")
async def get_profile(user: UserContext = Depends(get_current_user)):
    return {"user_id": user.user_id, "email": user.email}
```

---

## Running the API

```bash
# Development
uvicorn langscope.api.main:app --reload --port 8000

# Production
uvicorn langscope.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- GraphQL: http://localhost:8000/graphql

---

## API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with database status |
| `/models` | GET, POST, DELETE | Model CRUD operations |
| `/domains` | GET, POST, DELETE | Domain management |
| `/matches` | GET, POST | Match execution and history |
| `/leaderboard` | GET | Rankings by dimension |

### Evaluation Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ground-truth/domains` | GET | List GT evaluation domains |
| `/ground-truth/evaluate` | POST | Trigger GT evaluation |
| `/ground-truth/leaderboards/{domain}` | GET | GT domain rankings |
| `/prompts/classify` | POST | Classify prompt to domain |
| `/prompts/process` | POST | Classify + cache lookup |

### Session & Feedback

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/arena/sessions` | POST | Start arena session |
| `/arena/sessions/{id}/battles` | POST | Submit battle result |
| `/arena/sessions/{id}/complete` | POST | Complete session |
| `/recommendations` | GET | Get model recommendations |

### Infrastructure

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/cache/stats` | GET | Cache hit rates by category |
| `/cache/invalidate/{category}` | POST | Invalidate cache |
| `/params/{type}` | GET, PUT | Parameter management |
| `/monitoring/dashboard` | GET | System metrics |

### GraphQL

Full GraphQL API with:
- **47 types** (Model, Match, Rating, Domain, etc.)
- **Queries**: models, matches, leaderboard, groundTruthDomains
- **Mutations**: createModel, triggerMatch, invalidateCache
- **Subscriptions**: ratingUpdated, leaderboardUpdated

---

## Usage Examples

LangScope helps you find the best LLM for your specific use case. Here's how to use the API:

### Find the Best Model for Your Use Case

```bash
# Get top models for medical domain (best quality)
curl "http://localhost:8000/leaderboard/medical?dimension=raw_quality&limit=10"

# Get top models for coding (best value for money)
curl "http://localhost:8000/leaderboard/code_generation?dimension=cost_adjusted&limit=10"

# Get top models for Hindi language tasks
curl "http://localhost:8000/leaderboard/hindi_general?limit=10"
```

**Response:**
```json
{
  "domain": "medical",
  "dimension": "raw_quality",
  "entries": [
    {"rank": 1, "model_id": "gpt-4o", "name": "GPT-4o", "mu": 1687, "sigma": 42, "conservative": 1561},
    {"rank": 2, "model_id": "claude-3-opus", "name": "Claude 3 Opus", "mu": 1672, "sigma": 45, "conservative": 1537},
    {"rank": 3, "model_id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "mu": 1658, "sigma": 48, "conservative": 1514}
  ]
}
```

### Get Model Ratings Across Dimensions

```bash
# Get a model's rating in a specific domain (all 10 dimensions)
curl "http://localhost:8000/transfer/models/gpt-4o/ratings?domain=medical"
```

**Response:**
```json
{
  "model_id": "gpt-4o",
  "domain": "medical",
  "ratings": {
    "raw_quality": {"mu": 1687, "sigma": 42, "source": "direct"},
    "cost_adjusted": {"mu": 1542, "sigma": 55, "source": "direct"},
    "latency": {"mu": 1620, "sigma": 38, "source": "direct"},
    "hallucination_resistance": {"mu": 1701, "sigma": 51, "source": "direct"},
    "instruction_following": {"mu": 1695, "sigma": 44, "source": "direct"}
  }
}
```

### Compare Models for a Specific Task

```bash
# Compare multiple models in the legal domain
curl "http://localhost:8000/recommendations/compare?domain=legal&models=gpt-4o,claude-3-opus,gemini-1.5-pro"
```

### Get Personalized Recommendations

```bash
# Get recommendations based on your priorities
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "customer_support",
    "priorities": {
      "cost_adjusted": 0.4,
      "latency": 0.3,
      "raw_quality": 0.3
    },
    "max_cost_per_million": 5.0
  }'
```

### Find Similar Domains (Transfer Learning)

```bash
# If you need a model for a new domain, find similar evaluated domains
curl "http://localhost:8000/transfer/domains/bengali_medical/similar?limit=5"
```

**Response:**
```json
{
  "domain": "bengali_medical",
  "similar_domains": [
    {"domain": "hindi_medical", "correlation": 0.82, "facets": {"language": 0.55, "field": 1.0}},
    {"domain": "bengali_general", "correlation": 0.76, "facets": {"language": 1.0, "field": 0.5}},
    {"domain": "english_medical", "correlation": 0.71, "facets": {"language": 0.45, "field": 1.0}}
  ]
}
```

### Check Model Performance in Ground Truth Domains

```bash
# Get ASR (speech recognition) leaderboard
curl "http://localhost:8000/ground-truth/leaderboards/asr"

# Get code execution leaderboard (pass@k metrics)
curl "http://localhost:8000/ground-truth/leaderboards/code_execution"

# Get visual QA leaderboard
curl "http://localhost:8000/ground-truth/leaderboards/visual_qa"
```

### Python Client Example

```python
import httpx

BASE_URL = "http://localhost:8000"

# Find best model for your use case
def get_best_model(domain: str, dimension: str = "raw_quality") -> dict:
    """Get the top-ranked model for a domain."""
    response = httpx.get(f"{BASE_URL}/leaderboard/{domain}", params={
        "dimension": dimension,
        "limit": 1
    })
    entries = response.json()["entries"]
    return entries[0] if entries else None

# Example: Find best model for medical Q&A
best = get_best_model("medical", "raw_quality")
print(f"Best for medical: {best['name']} (μ={best['mu']:.0f})")

# Example: Find best value model for coding
best_value = get_best_model("code_generation", "cost_adjusted")
print(f"Best value for coding: {best_value['name']}")

# Example: Find fastest model for customer support
fastest = get_best_model("customer_support", "latency")
print(f"Fastest for support: {fastest['name']}")
```

### GraphQL Queries

```graphql
# Get leaderboard with specific dimensions
query {
  leaderboard(domain: "medical", dimension: RAW_QUALITY, limit: 5) {
    entries {
      rank
      modelId
      name
      mu
      sigma
      conservativeEstimate
    }
  }
}

# Get model rating with transfer learning
query {
  modelRating(modelId: "gpt-4o", domain: "hindi_medical", dimension: RAW_QUALITY) {
    mu
    sigma
    confidence
    source
    transferDetails {
      sourceDomains
      correlation
    }
  }
}
```

---

## Configuration

### TrueSkill Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `μ₀` | 1500.0 | Initial mean rating |
| `σ₀` | 166.0 | Initial uncertainty |
| `β` | 83.0 | Performance variability |
| `τ` | 8.3 | Dynamics factor (rating drift) |
| `k` | 3.0 | Conservative multiplier |

### Match Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `players_per_match` | 6 | Models per match (5-6 optimal) |
| `swiss_delta` | 75.0 | Max μ difference for grouping |
| `judge_count` | 3 | Number of judge models |

### Dimension Weights

Configurable per domain via `/params/dimension_weights`:

```python
# Example: Medical domain prioritizes hallucination resistance
{
    "raw_quality": 0.15,
    "hallucination_resistance": 0.25,
    "instruction_following": 0.20,
    # ... other dimensions
}
```

---

## Testing

```bash
# Run all tests
pytest test/ -v

# Run with coverage
pytest test/ -v --cov=langscope --cov-report=html

# Run specific modules
pytest test/test_ground_truth.py -v  # Ground truth tests
pytest test/test_cache.py -v          # Cache tests
pytest test/test_prompt.py -v         # Classification tests
pytest test/test_llm.py -v            # LLM provider tests
pytest test/test_ranking.py -v        # Ranking algorithm tests
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| API Routes | 94 | Endpoints, middleware, error handling |
| Auth | 26 | JWT validation, Supabase integration |
| Database | 85 | CRUD, ground truth, time series |
| Ground Truth | 78 | Judges, metrics, sampling, workflow |
| Cache | 45 | Sessions, rate limiting, semantic |
| Prompt | 32 | Classification, routing |
| Federation | 52 | Matches, judging, strata, workflow |
| Ranking | 68 | TrueSkill, Plackett-Luce, cost adjustment |
| Transfer | 87 | Faceted similarity, priors, correlation, specialists |
| Feedback | 41 | Arena, deltas, use-cases |
| LLM Providers | 48 | Message, Config, Response, all providers |
| Core | 98 | Models, ratings, dimensions, constants |

**Total: 754 tests | 51% code coverage**

### Key Module Coverage

| Module | Coverage |
|--------|----------|
| `ranking/plackett_luce.py` | 83% |
| `ranking/cost_adjustment.py` | 86% |
| `federation/strata.py` | 92% |
| `transfer/specialist.py` | 93% |
| `transfer/priors.py` | 88% |
| `transfer/transfer_learning.py` | 87% |

---

## Scripts

The `scripts/` folder contains operational utilities for setup and testing:

| Script | Description |
|--------|-------------|
| `seed_model_deployments.py` | Populate MongoDB with model data from Groq, OpenAI, XAI |
| `test_llm_providers.py` | Manual integration test for LLM API connections |
| `test_auth.py` | Manual test for Supabase authentication endpoints |

### Usage

```bash
# Seed model data (dry-run first)
python scripts/seed_model_deployments.py --dry-run
python scripts/seed_model_deployments.py --provider groq

# Test LLM provider connections
python scripts/test_llm_providers.py --provider openai --verbose

# Test authentication
python scripts/test_auth.py
```

**Note:** These scripts require actual API keys and running services, unlike the automated test suite.

---

## Architecture

```
langscope/
├── api/                    # REST API Layer
│   ├── routes/             # 18 route handlers
│   │   ├── arena.py        # User feedback sessions
│   │   ├── auth.py         # Authentication endpoints
│   │   ├── cache.py        # Cache management
│   │   ├── ground_truth.py # GT evaluation
│   │   ├── leaderboard.py  # Rankings
│   │   ├── matches.py      # Match execution
│   │   ├── prompts.py      # Domain classification
│   │   └── ...
│   ├── middleware/         # Auth (Supabase JWT), logging, rate limiting
│   └── schemas.py          # Pydantic models
│
├── llm/                    # LLM Provider Integration
│   ├── base.py             # BaseLLMProvider abstract class
│   ├── factory.py          # Provider factory
│   ├── models.py           # Message, LLMResponse, LLMConfig
│   └── providers/          # Provider implementations
│       ├── openai.py       # OpenAI (GPT-4, GPT-5)
│       ├── anthropic.py    # Anthropic (Claude)
│       ├── groq.py         # Groq (fast inference)
│       ├── xai.py          # XAI (Grok)
│       └── huggingface.py  # HuggingFace (Inference API + local)
│
├── models/                 # Data Models (see below)
│   ├── base/               # Base model definitions
│   ├── deployments/        # Cloud & self-hosted deployments
│   ├── benchmarks/         # External benchmarks & correlation
│   ├── sources/            # Data sources & sync automation
│   ├── timeseries/         # Rating, performance, price history
│   ├── reference/          # Hardware & quantization profiles
│   └── hashing.py          # Content & price hash utilities
│
├── core/                   # Core Data Structures
│   ├── rating.py           # TrueSkillRating
│   ├── model.py            # LLMModel
│   ├── dimensions.py       # 10 evaluation dimensions
│   └── deployment.py       # Multi-provider deployments (→ models/)
│
├── ranking/                # Rating Algorithms
│   ├── trueskill.py        # Multi-player TrueSkill
│   ├── plackett_luce.py    # Ranking probability
│   └── cost_adjustment.py  # Cost-adjusted scoring
│
├── federation/             # Peer Evaluation
│   ├── workflow.py         # Match orchestration
│   ├── selection.py        # Swiss pairing
│   ├── strata.py           # Role assignment
│   └── judge.py            # Ranking validation
│
├── ground_truth/           # Objective Evaluation
│   ├── workflow.py         # GT match workflow
│   ├── metrics.py          # WER, BLEU, pass@k
│   ├── judges/             # Domain-specific judges
│   │   ├── asr_judge.py    # Speech recognition
│   │   ├── tts_judge.py    # Text-to-speech
│   │   ├── code_judge.py   # Code execution
│   │   ├── visual_judge.py # Visual QA
│   │   └── needle_judge.py # Long context retrieval
│   └── samples/            # Sample datasets
│
├── prompt/                 # Domain Classification
│   ├── classifier.py       # Hierarchical classifier
│   ├── manager.py          # Prompt processing
│   └── constants.py        # Categories, patterns
│
├── cache/                  # Multi-Layer Caching
│   ├── manager.py          # Unified cache manager
│   ├── categories.py       # Cache configuration
│   ├── semantic_cache.py   # Qdrant integration
│   └── rate_limit.py       # Sliding window limiter
│
├── transfer/               # Cross-Domain Transfer
│   ├── faceted.py          # Multi-faceted domain similarity
│   ├── priors.py           # Expert priors (125+ pairs)
│   ├── correlation.py      # Bayesian correlation
│   └── specialist.py       # Specialist detection
│
├── feedback/               # User Feedback
│   ├── workflow.py         # Arena sessions
│   ├── delta.py            # Rating deltas
│   └── use_case.py         # Use-case adjustments
│
├── database/               # MongoDB Integration
│   ├── mongodb.py          # 50+ database operations
│   └── schemas.py          # Document validation
│
└── graphql/                # GraphQL API
    ├── types.py            # 47 GraphQL types
    ├── queries.py          # Query resolvers
    ├── mutations.py        # Mutation resolvers
    └── subscriptions.py    # Real-time updates
```

### Data Models Package (`langscope/models/`)

The `models` package consolidates all model-related data structures for multi-provider LLM deployments, external benchmarks, and automation.

```
langscope/models/
├── base/                   # Base Model Definitions
│   ├── model.py           # BaseModel, BenchmarkScore
│   ├── architecture.py    # Architecture, ArchitectureType (MoE, etc.)
│   ├── capabilities.py    # Modalities, language support, features
│   ├── context.py         # Context window configuration
│   ├── license.py         # License information
│   └── quantization.py    # Quantization options (AWQ, GPTQ, etc.)
│
├── deployments/            # Provider Deployments
│   ├── cloud.py           # ModelDeployment, Provider, Pricing
│   └── self_hosted.py     # SelfHostedDeployment, HardwareConfig
│
├── benchmarks/             # External Benchmarks
│   ├── definitions.py     # MMLU, HumanEval, Arena definitions
│   ├── results.py         # BenchmarkResult, aggregates
│   └── correlation.py     # Correlation with LangScope ratings
│
├── sources/                # Data Source Automation
│   ├── data_sources.py    # OpenRouter, LMSYS, leaderboard sources
│   └── sync.py            # Sync engine for automated updates
│
├── timeseries/             # Time Series Data
│   ├── ratings.py         # Rating history snapshots
│   ├── performance.py     # Per-match performance metrics
│   └── prices.py          # Price change history
│
├── reference/              # Reference Data
│   ├── hardware.py        # GPU profiles (A100, H100, etc.)
│   └── quantization.py    # Quantization method profiles
│
└── hashing.py              # Hash utilities for deduplication
```

**Usage:**

```python
# Recommended imports (new)
from langscope.models import BaseModel, ModelDeployment, BenchmarkDefinition
from langscope.models import content_hash, price_hash, PREDEFINED_BENCHMARKS

# Create a base model
model = BaseModel(
    id="meta-llama/llama-3.1-70b",
    name="Llama 3.1 70B",
    family="llama",
    organization="Meta"
)
model.add_quantization("awq-4bit", bits=4, vram_gb=40, quality_retention=0.95)
model.add_benchmark("mmlu", score=82.5, variant="5-shot")

# Create a deployment
deployment = ModelDeployment(
    id="groq/llama-3.1-70b-versatile",
    base_model_id="meta-llama/llama-3.1-70b",
    provider=Provider(id="groq", name="Groq")
)
deployment.pricing.input_cost_per_million = 0.59
deployment.pricing.output_cost_per_million = 0.79
```

**Backward Compatibility:**

Old imports continue to work:

```python
from langscope.core.base_model import BaseModel  # Still works
from langscope.core.deployment import ModelDeployment  # Still works
```

---

## License

See LICENSE file for details.

## Citation

```bibtex
@software{langscope2024,
  title={LangScope: Multi-domain LLM Evaluation with TrueSkill and Plackett-Luce},
  author={Sourav Bandyopadhyay},
  year={2024},
  url={https://github.com/souravbandyo/langscope-algorithm}
}
```

## References

- Herbrich, R., Minka, T., & Graepel, T. (2006). TrueSkill: A Bayesian skill rating system.
- Plackett, R. L. (1975). The analysis of permutations.
- Luce, R. D. (1959). Individual choice behavior.

---

**Requirements:** Python 3.9+, MongoDB  
**Optional:** Redis (caching), Qdrant (semantic search), sentence-transformers (classification)  
**Documentation:** See `env.template` for configuration details.
