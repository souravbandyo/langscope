"""
Needle in Haystack Analytics.

Provides visualization and analysis tools for needle in haystack
evaluation results, including accuracy heatmaps.
"""

from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langscope.database.mongodb import MongoDB


def compute_accuracy_heatmap(
    db: 'MongoDB',
    model_id: str,
    domain: str = "needle_in_haystack"
) -> Dict[str, Dict[str, float]]:
    """
    Compute accuracy heatmap for a model.
    
    Args:
        db: Database instance
        model_id: Model identifier
        domain: Domain name
    
    Returns:
        Heatmap: {context_length: {needle_position: accuracy}}
    """
    if not db or not db.connected:
        return {}
    
    # Query matches for this model
    matches = list(db.db["ground_truth_matches"].find({
        "domain": domain,
        f"scores.{model_id}": {"$exists": True}
    }))
    
    heatmap: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    
    for match in matches:
        scores = match.get("scores", {}).get(model_id, {})
        metadata = match.get("sample_metadata", {})
        
        ctx_len = str(metadata.get("context_length", 0))
        needle_pos = str(metadata.get("needle_position", 0.5))
        
        metrics = scores.get("metrics", {})
        accuracy = metrics.get("retrieval_accuracy", 0.0)
        
        if ctx_len not in heatmap:
            heatmap[ctx_len] = {}
            counts[ctx_len] = {}
        
        if needle_pos not in heatmap[ctx_len]:
            heatmap[ctx_len][needle_pos] = 0.0
            counts[ctx_len][needle_pos] = 0
        
        heatmap[ctx_len][needle_pos] += accuracy
        counts[ctx_len][needle_pos] += 1
    
    # Compute averages
    for ctx_len in heatmap:
        for needle_pos in heatmap[ctx_len]:
            count = counts[ctx_len][needle_pos]
            if count > 0:
                heatmap[ctx_len][needle_pos] /= count
    
    return heatmap


def get_model_performance_summary(
    db: 'MongoDB',
    model_id: str,
    domain: str = "needle_in_haystack"
) -> Dict[str, Any]:
    """
    Get performance summary for a model.
    
    Args:
        db: Database instance
        model_id: Model identifier
        domain: Domain name
    
    Returns:
        Summary statistics
    """
    if not db or not db.connected:
        return {}
    
    # Query matches for this model
    matches = list(db.db["ground_truth_matches"].find({
        "domain": domain,
        f"scores.{model_id}": {"$exists": True}
    }))
    
    if not matches:
        return {
            "model_id": model_id,
            "total_samples": 0,
            "avg_accuracy": 0.0,
        }
    
    # Aggregate statistics
    total_accuracy = 0.0
    by_length: Dict[int, List[float]] = {}
    by_position: Dict[float, List[float]] = {}
    
    for match in matches:
        scores = match.get("scores", {}).get(model_id, {})
        metadata = match.get("sample_metadata", {})
        
        ctx_len = metadata.get("context_length", 0)
        needle_pos = metadata.get("needle_position", 0.5)
        
        metrics = scores.get("metrics", {})
        accuracy = metrics.get("retrieval_accuracy", 0.0)
        
        total_accuracy += accuracy
        
        if ctx_len not in by_length:
            by_length[ctx_len] = []
        by_length[ctx_len].append(accuracy)
        
        if needle_pos not in by_position:
            by_position[needle_pos] = []
        by_position[needle_pos].append(accuracy)
    
    # Compute averages
    avg_by_length = {
        k: sum(v) / len(v) if v else 0.0
        for k, v in by_length.items()
    }
    
    avg_by_position = {
        k: sum(v) / len(v) if v else 0.0
        for k, v in by_position.items()
    }
    
    # Find best and worst
    best_length = max(avg_by_length.keys(), key=lambda k: avg_by_length[k], default=0)
    worst_length = min(avg_by_length.keys(), key=lambda k: avg_by_length[k], default=0)
    
    best_position = max(avg_by_position.keys(), key=lambda k: avg_by_position[k], default=0.5)
    worst_position = min(avg_by_position.keys(), key=lambda k: avg_by_position[k], default=0.5)
    
    return {
        "model_id": model_id,
        "total_samples": len(matches),
        "avg_accuracy": total_accuracy / len(matches) if matches else 0.0,
        "accuracy_by_length": avg_by_length,
        "accuracy_by_position": avg_by_position,
        "best_length": best_length,
        "worst_length": worst_length,
        "best_position": best_position,
        "worst_position": worst_position,
        "perfect_retrieval": all(
            avg >= 0.99 for avg in avg_by_length.values()
        ) if avg_by_length else False,
    }


def compare_models_at_depth(
    db: 'MongoDB',
    model_ids: List[str],
    context_length: int,
    domain: str = "needle_in_haystack"
) -> Dict[str, Dict[str, float]]:
    """
    Compare models at a specific context length.
    
    Args:
        db: Database instance
        model_ids: List of model IDs to compare
        context_length: Context length to analyze
        domain: Domain name
    
    Returns:
        {model_id: {needle_position: accuracy}}
    """
    if not db or not db.connected:
        return {}
    
    # Query matches at this context length
    matches = list(db.db["ground_truth_matches"].find({
        "domain": domain,
        "sample_metadata.context_length": context_length,
    }))
    
    results: Dict[str, Dict[str, List[float]]] = {
        mid: {} for mid in model_ids
    }
    
    for match in matches:
        metadata = match.get("sample_metadata", {})
        needle_pos = str(metadata.get("needle_position", 0.5))
        
        for model_id in model_ids:
            scores = match.get("scores", {}).get(model_id)
            if not scores:
                continue
            
            metrics = scores.get("metrics", {})
            accuracy = metrics.get("retrieval_accuracy", 0.0)
            
            if needle_pos not in results[model_id]:
                results[model_id][needle_pos] = []
            results[model_id][needle_pos].append(accuracy)
    
    # Compute averages
    comparison = {}
    for model_id, positions in results.items():
        comparison[model_id] = {
            pos: sum(acc) / len(acc) if acc else 0.0
            for pos, acc in positions.items()
        }
    
    return comparison


