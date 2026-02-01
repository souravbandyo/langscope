"""
Hierarchical Domain Classifier for LangScope.

Two-stage classification: Category -> Base Domain -> Language Variant.
"""

import re
import logging
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np

from langscope.prompt.constants import (
    CATEGORIES,
    DOMAIN_KEYWORDS,
    LANGUAGE_PATTERNS,
    GROUND_TRUTH_DOMAINS,
    map_to_template_name,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of domain classification."""
    
    # Hierarchy levels
    category: str           # "core_language", "multimodal", etc.
    base_domain: str        # "medical", "asr", etc.
    variant: Optional[str]  # "hindi", "odia", None for English
    
    # Confidence
    confidence: float       # 0.0 - 1.0
    
    # Routing information
    is_ground_truth: bool   # Route to GT evaluation?
    
    # Additional context
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    
    @property
    def full_domain_name(self) -> str:
        """Get full domain name with variant prefix if applicable."""
        if self.variant and self.variant != "en":
            return f"{self.variant}_{self.base_domain}"
        return self.base_domain
    
    @property
    def template_name(self) -> str:
        """Get mapped template name for DomainManager."""
        return map_to_template_name(self.full_domain_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "base_domain": self.base_domain,
            "variant": self.variant,
            "confidence": self.confidence,
            "is_ground_truth": self.is_ground_truth,
            "full_domain_name": self.full_domain_name,
            "template_name": self.template_name,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassificationResult':
        """Create from dictionary."""
        return cls(
            category=data["category"],
            base_domain=data["base_domain"],
            variant=data.get("variant"),
            confidence=data["confidence"],
            is_ground_truth=data["is_ground_truth"],
        )


class HierarchicalDomainClassifier:
    """
    Two-stage hierarchical classifier for domain routing.
    
    Stage 1: Classify into category (7 classes)
    Stage 2: Classify into base domain within category
    Stage 3: Detect language variant (regex-based)
    
    Uses sentence embeddings for similarity-based classification.
    Total time: ~12-15ms on CPU
    """
    
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIM = 384
    
    def __init__(
        self,
        model_name: str = None,
        centroids_path: str = None,
        lazy_load: bool = True,
    ):
        """
        Initialize classifier.
        
        Args:
            model_name: Sentence transformer model name
            centroids_path: Path to pre-trained centroids pickle
            lazy_load: If True, delay loading model until first use
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._encoder = None
        self._lazy_load = lazy_load
        
        # Centroids for each category/domain
        self.category_centroids: Dict[str, np.ndarray] = {}
        self.domain_centroids: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Keyword lists
        self.domain_keywords = DOMAIN_KEYWORDS
        
        # Load centroids if path provided
        if centroids_path:
            self.load_centroids(centroids_path)
        
        # Load model immediately if not lazy
        if not lazy_load:
            self._load_encoder()
    
    @property
    def encoder(self):
        """Lazy-loaded sentence transformer encoder."""
        if self._encoder is None:
            self._load_encoder()
        return self._encoder
    
    def _load_encoder(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name)
            logger.info(f"Loaded encoder: {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
    
    def classify(self, prompt: str) -> ClassificationResult:
        """
        Classify prompt into domain hierarchy.
        
        Total time: ~12-15ms on CPU
        
        Args:
            prompt: Input prompt text
        
        Returns:
            ClassificationResult with category, domain, variant
        """
        # Stage 1: Language detection (regex, ~0.1ms)
        language = self._detect_language(prompt)
        
        # Stage 2: Get embedding (~10ms)
        embedding = self.encoder.encode(prompt, convert_to_numpy=True)
        
        # Stage 3: Category classification (~1ms)
        category, cat_conf = self._classify_category(embedding, prompt)
        
        # Stage 4: Domain classification within category (~1ms)
        base_domain, domain_conf = self._classify_domain(
            embedding, prompt, category
        )
        
        # Stage 5: Check if ground truth domain
        is_gt = base_domain in GROUND_TRUTH_DOMAINS
        
        # Combined confidence
        confidence = min(cat_conf, domain_conf)
        
        return ClassificationResult(
            category=category,
            base_domain=base_domain,
            variant=language,
            confidence=float(confidence),
            is_ground_truth=is_gt,
            embedding=embedding,
        )
    
    def classify_batch(self, prompts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple prompts efficiently.
        
        Args:
            prompts: List of prompt texts
        
        Returns:
            List of ClassificationResult
        """
        # Batch encode for efficiency
        embeddings = self.encoder.encode(prompts, convert_to_numpy=True)
        
        results = []
        for prompt, embedding in zip(prompts, embeddings):
            # Language detection
            language = self._detect_language(prompt)
            
            # Category classification
            category, cat_conf = self._classify_category(embedding, prompt)
            
            # Domain classification
            base_domain, domain_conf = self._classify_domain(
                embedding, prompt, category
            )
            
            # Ground truth check
            is_gt = base_domain in GROUND_TRUTH_DOMAINS
            
            results.append(ClassificationResult(
                category=category,
                base_domain=base_domain,
                variant=language,
                confidence=float(min(cat_conf, domain_conf)),
                is_ground_truth=is_gt,
                embedding=embedding,
            ))
        
        return results
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text (useful for caching).
        
        Args:
            text: Input text
        
        Returns:
            384-dimensional embedding vector
        """
        return self.encoder.encode(text, convert_to_numpy=True)
    
    def _detect_language(self, text: str) -> Optional[str]:
        """
        Fast regex-based language detection (~0.1ms).
        
        Args:
            text: Input text
        
        Returns:
            Language code or None for English
        """
        for lang, pattern in LANGUAGE_PATTERNS.items():
            if re.search(pattern, text):
                return lang
        return None
    
    def _classify_category(
        self,
        embedding: np.ndarray,
        prompt: str,
    ) -> Tuple[str, float]:
        """
        Classify into top-level category.
        
        Args:
            embedding: Prompt embedding
            prompt: Original prompt text
        
        Returns:
            (category_name, confidence)
        """
        # If no centroids trained, use keyword matching as fallback
        if not self.category_centroids:
            return self._keyword_category_fallback(prompt)
        
        similarities = {}
        prompt_lower = prompt.lower()
        
        for category, centroid in self.category_centroids.items():
            sim = self._cosine_similarity(embedding, centroid)
            
            # Keyword boosting for domains in this category
            for domain in CATEGORIES.get(category, []):
                keywords = self.domain_keywords.get(domain, [])
                if any(kw.lower() in prompt_lower for kw in keywords):
                    sim *= 1.15  # 15% boost
                    break
            
            similarities[category] = sim
        
        best = max(similarities, key=similarities.get)
        return best, similarities[best]
    
    def _classify_domain(
        self,
        embedding: np.ndarray,
        prompt: str,
        category: str,
    ) -> Tuple[str, float]:
        """
        Classify into base domain within category.
        
        Args:
            embedding: Prompt embedding
            prompt: Original prompt text
            category: Category from stage 1
        
        Returns:
            (domain_name, confidence)
        """
        # If no centroids for category, use keyword matching
        if category not in self.domain_centroids:
            return self._keyword_domain_fallback(prompt, category)
        
        similarities = {}
        prompt_lower = prompt.lower()
        
        for domain, centroid in self.domain_centroids[category].items():
            sim = self._cosine_similarity(embedding, centroid)
            
            # Keyword boosting
            keywords = self.domain_keywords.get(domain, [])
            if any(kw.lower() in prompt_lower for kw in keywords):
                sim *= 1.2  # 20% boost for domain-specific keywords
            
            similarities[domain] = sim
        
        best = max(similarities, key=similarities.get)
        return best, similarities[best]
    
    def _keyword_category_fallback(self, prompt: str) -> Tuple[str, float]:
        """
        Fallback category detection using keywords.
        
        Used when no centroids are trained.
        """
        prompt_lower = prompt.lower()
        scores = {cat: 0.0 for cat in CATEGORIES}
        
        for category, domains in CATEGORIES.items():
            for domain in domains:
                keywords = self.domain_keywords.get(domain, [])
                matches = sum(1 for kw in keywords if kw.lower() in prompt_lower)
                scores[category] += matches
        
        best = max(scores, key=scores.get)
        # Normalize confidence
        total = sum(scores.values())
        confidence = scores[best] / total if total > 0 else 0.5
        
        return best, min(confidence + 0.5, 1.0)  # Base confidence of 0.5
    
    def _keyword_domain_fallback(
        self,
        prompt: str,
        category: str,
    ) -> Tuple[str, float]:
        """
        Fallback domain detection using keywords.
        
        Used when no centroids are trained for the category.
        """
        prompt_lower = prompt.lower()
        domains = CATEGORIES.get(category, [])
        
        if not domains:
            return "general", 0.5
        
        scores = {}
        for domain in domains:
            keywords = self.domain_keywords.get(domain, [])
            matches = sum(1 for kw in keywords if kw.lower() in prompt_lower)
            scores[domain] = matches
        
        best = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[best] / total if total > 0 else 0.5
        
        return best, min(confidence + 0.5, 1.0)
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    # =========================================================================
    # Training
    # =========================================================================
    
    def train_centroids(
        self,
        training_data: Dict[str, Dict[str, List[str]]],
    ) -> Dict[str, Any]:
        """
        Train centroids from labeled examples.
        
        Args:
            training_data: {category: {domain: [example_prompts]}}
        
        Returns:
            Training statistics
        """
        stats = {
            "categories": 0,
            "domains": 0,
            "examples": 0,
        }
        
        for category, domains in training_data.items():
            all_embeddings = []
            self.domain_centroids[category] = {}
            
            for domain, examples in domains.items():
                if not examples:
                    continue
                
                # Encode all examples for this domain
                embeddings = self.encoder.encode(examples, convert_to_numpy=True)
                
                # Compute centroid (mean embedding)
                centroid = embeddings.mean(axis=0)
                self.domain_centroids[category][domain] = centroid
                
                all_embeddings.extend(embeddings)
                stats["domains"] += 1
                stats["examples"] += len(examples)
            
            # Compute category centroid from all domain embeddings
            if all_embeddings:
                self.category_centroids[category] = np.array(all_embeddings).mean(axis=0)
                stats["categories"] += 1
        
        logger.info(f"Trained centroids: {stats}")
        return stats
    
    def save_centroids(self, path: str) -> None:
        """
        Save trained centroids to disk.
        
        Args:
            path: Output file path (pickle format)
        """
        data = {
            "category_centroids": self.category_centroids,
            "domain_centroids": self.domain_centroids,
            "model_name": self.model_name,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved centroids to {path}")
    
    def load_centroids(self, path: str) -> bool:
        """
        Load trained centroids from disk.
        
        Args:
            path: Input file path (pickle format)
        
        Returns:
            True if loaded successfully
        """
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            self.category_centroids = data.get("category_centroids", {})
            self.domain_centroids = data.get("domain_centroids", {})
            
            logger.info(f"Loaded centroids from {path}: {len(self.category_centroids)} categories")
            return True
        except FileNotFoundError:
            logger.warning(f"Centroids file not found: {path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load centroids: {e}")
            return False
    
    def evaluate(
        self,
        test_data: Dict[str, Dict[str, List[str]]],
    ) -> Dict[str, Any]:
        """
        Evaluate classifier accuracy on test data.
        
        Args:
            test_data: {category: {domain: [example_prompts]}}
        
        Returns:
            Evaluation metrics
        """
        results = {
            "category_accuracy": 0.0,
            "domain_accuracy": 0.0,
            "total": 0,
            "category_correct": 0,
            "domain_correct": 0,
            "confusion_matrix": {},
        }
        
        for category, domains in test_data.items():
            for domain, examples in domains.items():
                for example in examples:
                    result = self.classify(example)
                    results["total"] += 1
                    
                    if result.category == category:
                        results["category_correct"] += 1
                    
                    if result.base_domain == domain:
                        results["domain_correct"] += 1
                    else:
                        # Track confusion
                        key = f"{domain} -> {result.base_domain}"
                        results["confusion_matrix"][key] = \
                            results["confusion_matrix"].get(key, 0) + 1
        
        if results["total"] > 0:
            results["category_accuracy"] = results["category_correct"] / results["total"]
            results["domain_accuracy"] = results["domain_correct"] / results["total"]
        
        return results

