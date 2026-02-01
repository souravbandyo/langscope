"""
Benchmarks API Routes.

Provides endpoints for:
- External benchmark definitions (MMLU, HumanEval, etc.)
- Benchmark results for base models
- Benchmark correlations with LangScope ratings
"""

from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from langscope.api.dependencies import get_db


router = APIRouter(
    prefix="/benchmarks",
    tags=["benchmarks"],
    responses={404: {"description": "Not found"}},
)


# === Pydantic Schemas ===

class BenchmarkScoring(BaseModel):
    """Scoring methodology for a benchmark."""
    method: str = Field(..., description="Scoring method (accuracy, pass@k, etc.)")
    min_value: float = Field(0, description="Minimum score")
    max_value: float = Field(100, description="Maximum score")
    higher_is_better: bool = Field(True, description="Whether higher is better")


class BenchmarkDefinitionResponse(BaseModel):
    """Benchmark definition response."""
    id: str = Field(..., alias="_id")
    name: str
    category: str
    description: str
    what_it_tests: List[str]
    scoring: BenchmarkScoring
    source_url: str
    automation_available: bool = False
    created_at: datetime
    
    class Config:
        populate_by_name = True


class BenchmarkResultResponse(BaseModel):
    """Benchmark result for a model."""
    benchmark_id: str
    base_model_id: str
    score: float
    variant: str = ""
    source_url: str = ""
    measured_at: datetime
    verified: bool = False


class CreateBenchmarkDefinitionRequest(BaseModel):
    """Request to create a new benchmark definition."""
    id: str = Field(..., description="Unique benchmark ID (e.g., 'mmlu', 'humaneval')")
    name: str = Field(..., description="Human-readable name")
    category: str = Field(..., description="Category (reasoning, coding, etc.)")
    description: str = Field(..., description="What the benchmark measures")
    what_it_tests: List[str] = Field(default_factory=list, description="Specific capabilities tested")
    scoring: BenchmarkScoring
    source_url: str = Field("", description="Official source URL")


class CreateBenchmarkResultRequest(BaseModel):
    """Request to create a benchmark result."""
    benchmark_id: str
    base_model_id: str
    score: float
    variant: str = Field("", description="Score variant (e.g., '5-shot', 'zero-shot')")
    source_url: str = Field("", description="Source of this score")


class CorrelationResponse(BaseModel):
    """Correlation between benchmark and LangScope rating."""
    benchmark_id: str
    langscope_dimension: str
    domain: str
    correlation: float = Field(..., ge=-1, le=1)
    p_value: float
    sample_size: int
    computed_at: datetime


# === Endpoints ===

@router.get("/definitions", response_model=List[BenchmarkDefinitionResponse])
def list_benchmark_definitions(
    category: Optional[str] = Query(None, description="Filter by category"),
    db=Depends(get_db),
):
    """
    List all benchmark definitions.
    
    Optionally filter by category (reasoning, coding, language, safety, etc.)
    """
    filters = {}
    if category:
        filters["category"] = category
    
    benchmarks = db.get_all_benchmark_definitions(filters=filters)
    return benchmarks


@router.get("/definitions/{benchmark_id}", response_model=BenchmarkDefinitionResponse)
def get_benchmark_definition(
    benchmark_id: str,
    db=Depends(get_db),
):
    """Get a specific benchmark definition."""
    benchmark = db.get_benchmark_definition(benchmark_id)
    if not benchmark:
        raise HTTPException(status_code=404, detail=f"Benchmark {benchmark_id} not found")
    return benchmark


@router.post("/definitions", response_model=BenchmarkDefinitionResponse, status_code=201)
def create_benchmark_definition(
    request: CreateBenchmarkDefinitionRequest,
    db=Depends(get_db),
):
    """Create a new benchmark definition."""
    # Check for duplicates
    existing = db.get_benchmark_definition(request.id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Benchmark {request.id} already exists")
    
    now = datetime.utcnow()
    benchmark = {
        "_id": request.id,
        "name": request.name,
        "category": request.category,
        "description": request.description,
        "what_it_tests": request.what_it_tests,
        "scoring": request.scoring.model_dump(),
        "source_url": request.source_url,
        "automation_available": False,
        "created_at": now,
        "updated_at": now,
    }
    
    db.save_benchmark_definition(benchmark)
    return benchmark


@router.delete("/definitions/{benchmark_id}")
def delete_benchmark_definition(
    benchmark_id: str,
    db=Depends(get_db),
):
    """Delete a benchmark definition."""
    deleted = db.delete_benchmark_definition(benchmark_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Benchmark {benchmark_id} not found")
    return {"message": f"Benchmark {benchmark_id} deleted"}


# === Benchmark Results ===

@router.get("/results/{base_model_id}", response_model=List[BenchmarkResultResponse])
def get_model_benchmark_results(
    base_model_id: str,
    benchmark_id: Optional[str] = Query(None, description="Filter by benchmark"),
    db=Depends(get_db),
):
    """
    Get all benchmark results for a base model.
    
    Optionally filter by specific benchmark.
    """
    filters = {"base_model_id": base_model_id}
    if benchmark_id:
        filters["benchmark_id"] = benchmark_id
    
    results = db.get_benchmark_results(filters=filters)
    return results


@router.post("/results", response_model=BenchmarkResultResponse, status_code=201)
def create_benchmark_result(
    request: CreateBenchmarkResultRequest,
    db=Depends(get_db),
):
    """Submit a benchmark result for a model."""
    # Verify benchmark exists
    benchmark = db.get_benchmark_definition(request.benchmark_id)
    if not benchmark:
        raise HTTPException(status_code=404, detail=f"Benchmark {request.benchmark_id} not found")
    
    # Verify base model exists
    base_model = db.get_base_model(request.base_model_id)
    if not base_model:
        raise HTTPException(status_code=404, detail=f"Base model {request.base_model_id} not found")
    
    now = datetime.utcnow()
    result = {
        "benchmark_id": request.benchmark_id,
        "base_model_id": request.base_model_id,
        "score": request.score,
        "variant": request.variant,
        "source_url": request.source_url,
        "measured_at": now,
        "verified": False,
        "created_at": now,
    }
    
    db.save_benchmark_result(result)
    return result


@router.get("/compare", response_model=List[dict])
def compare_benchmarks(
    base_model_ids: List[str] = Query(..., description="List of base model IDs to compare"),
    benchmark_id: Optional[str] = Query(None, description="Filter by benchmark"),
    db=Depends(get_db),
):
    """
    Compare benchmark scores across multiple base models.
    
    Returns a matrix of scores for easy comparison.
    """
    comparison = []
    
    for model_id in base_model_ids:
        model = db.get_base_model(model_id)
        if not model:
            continue
        
        filters = {"base_model_id": model_id}
        if benchmark_id:
            filters["benchmark_id"] = benchmark_id
        
        results = db.get_benchmark_results(filters=filters)
        
        scores = {r["benchmark_id"]: r["score"] for r in results}
        
        comparison.append({
            "base_model_id": model_id,
            "name": model.get("name", model_id),
            "organization": model.get("organization", ""),
            "scores": scores,
        })
    
    return comparison


# === Correlations ===

@router.get("/correlations", response_model=List[CorrelationResponse])
def get_benchmark_correlations(
    benchmark_id: Optional[str] = Query(None, description="Filter by benchmark"),
    dimension: Optional[str] = Query(None, description="Filter by LangScope dimension"),
    min_correlation: float = Query(0.0, ge=-1, le=1, description="Minimum correlation"),
    db=Depends(get_db),
):
    """
    Get correlations between external benchmarks and LangScope ratings.
    
    Useful for understanding how external benchmarks predict LangScope performance.
    """
    filters = {}
    if benchmark_id:
        filters["benchmark_id"] = benchmark_id
    if dimension:
        filters["langscope_dimension"] = dimension
    
    correlations = db.get_benchmark_correlations(filters=filters)
    
    # Filter by minimum correlation
    filtered = [c for c in correlations if abs(c.get("correlation", 0)) >= min_correlation]
    
    return filtered


@router.get("/leaderboard/{benchmark_id}", response_model=List[dict])
def get_benchmark_leaderboard(
    benchmark_id: str,
    variant: Optional[str] = Query(None, description="Score variant filter"),
    limit: int = Query(50, ge=1, le=100),
    db=Depends(get_db),
):
    """
    Get a leaderboard for a specific benchmark.
    
    Returns models ranked by their score on this benchmark.
    """
    benchmark = db.get_benchmark_definition(benchmark_id)
    if not benchmark:
        raise HTTPException(status_code=404, detail=f"Benchmark {benchmark_id} not found")
    
    filters = {"benchmark_id": benchmark_id}
    if variant:
        filters["variant"] = variant
    
    # Get all results for this benchmark
    results = db.get_benchmark_results(filters=filters)
    
    # Sort by score (respecting higher_is_better)
    scoring = benchmark.get("scoring", {})
    higher_is_better = scoring.get("higher_is_better", True)
    
    results.sort(key=lambda x: x.get("score", 0), reverse=higher_is_better)
    
    # Build leaderboard
    leaderboard = []
    for rank, result in enumerate(results[:limit], 1):
        model_id = result.get("base_model_id")
        model = db.get_base_model(model_id)
        
        leaderboard.append({
            "rank": rank,
            "base_model_id": model_id,
            "name": model.get("name", model_id) if model else model_id,
            "organization": model.get("organization", "") if model else "",
            "score": result.get("score"),
            "variant": result.get("variant", ""),
            "source_url": result.get("source_url", ""),
        })
    
    return leaderboard

