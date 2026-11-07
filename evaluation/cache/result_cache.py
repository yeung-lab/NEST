# evaluation/cache/result_cache.py
"""Result caching system for evaluation responses and consistency checking"""

import json
import hashlib
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Single cache entry for a model response"""
    question_id: str
    model_id: str
    prompt: str
    media_path: Optional[str]
    response: str
    timestamp: str
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary"""
        return cls(**data)

class ResultCache:
    """Cache system for storing and retrieving model responses"""
    
    def __init__(self, cache_dir: Union[str, os.PathLike] = "evaluation_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "responses.json")
        self.consistency_file = os.path.join(self.cache_dir, "consistency.json")
        self.failures_file = os.path.join(self.cache_dir, "failures.json")

        # Load existing cache
        self._load_cache()
    
    def _load_cache(self):
        """Load existing cache from disk"""
        self.cache: Dict[str, CacheEntry] = {}
        self.consistency_cache: Dict[str, List[CacheEntry]] = {}
        self.failures: Dict[str, Dict[str, Any]] = {}  # key -> {count, last_error, timestamp}

        # Load responses cache
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        self.cache[key] = CacheEntry.from_dict(entry_data)
                logger.info(f"Loaded {len(self.cache)} cached responses")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # Load consistency cache
        if os.path.exists(self.consistency_file):
            try:
                with open(self.consistency_file, 'r') as f:
                    data = json.load(f)
                    for key, entries_data in data.items():
                        self.consistency_cache[key] = [
                            CacheEntry.from_dict(entry_data)
                            for entry_data in entries_data
                        ]
                logger.info(f"Loaded {len(self.consistency_cache)} consistency groups")
            except Exception as e:
                logger.warning(f"Failed to load consistency cache: {e}")

        # Load failure cache
        if os.path.exists(self.failures_file):
            try:
                with open(self.failures_file, 'r') as f:
                    self.failures = json.load(f)
                n_over = sum(1 for v in self.failures.values() if v.get("count", 0) >= 3)
                logger.info(f"Loaded {len(self.failures)} failure records ({n_over} exceeded retry limit)")
            except Exception as e:
                logger.warning(f"Failed to load failure cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            # Save responses cache
            cache_data = {key: entry.to_dict() for key, entry in self.cache.items()}
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            # Save consistency cache
            consistency_data = {
                key: [entry.to_dict() for entry in entries]
                for key, entries in self.consistency_cache.items()
            }
            with open(self.consistency_file, 'w') as f:
                json.dump(consistency_data, f, indent=2)

            # Save failure cache
            if self.failures:
                with open(self.failures_file, 'w') as f:
                    json.dump(self.failures, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _generate_key(self, 
                     question_id: str, 
                     model_id: str, 
                     prompt: str, 
                     media_path: Optional[str] = None) -> str:
        """Generate cache key for a request"""
        # Create hash of the request parameters
        content = f"{question_id}:{model_id}:{prompt}:{media_path or ''}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_response(self,
                    question_id: str,
                    model_id: str,
                    prompt: str,
                    media_path: Optional[str] = None) -> Optional[Tuple[str, Optional[str]]]:
        """Get cached response if available

        Returns:
            Tuple of (response, reasoning) if cache hit, None if miss
        """
        key = self._generate_key(question_id, model_id, prompt, media_path)
        entry = self.cache.get(key)

        if entry:
            logger.debug(f"Cache hit for question {question_id}")
            return (entry.response, entry.reasoning)

        return None
    
    def store_response(self,
                      question_id: str,
                      model_id: str,
                      prompt: str,
                      response: str,
                      media_path: Optional[str] = None,
                      reasoning: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store response in cache"""
        key = self._generate_key(question_id, model_id, prompt, media_path)

        entry = CacheEntry(
            question_id=question_id,
            model_id=model_id,
            prompt=prompt,
            media_path=media_path,
            response=response,
            timestamp=datetime.now().isoformat(),
            reasoning=reasoning,
            metadata=metadata
        )

        self.cache[key] = entry
        self._save_cache()
        logger.debug(f"Cached response for question {question_id}")
    
    def record_failure(self,
                      question_id: str,
                      model_id: str,
                      prompt: str,
                      error: str,
                      media_path: Optional[str] = None) -> int:
        """Record a failed API call. Returns the new failure count."""
        key = self._generate_key(question_id, model_id, prompt, media_path)
        entry = self.failures.get(key, {"question_id": question_id, "count": 0})
        entry["count"] += 1
        entry["last_error"] = error
        entry["timestamp"] = datetime.now().isoformat()
        entry["question_id"] = question_id
        self.failures[key] = entry
        self._save_cache()
        return entry["count"]

    def get_failure_count(self,
                         question_id: str,
                         model_id: str,
                         prompt: str,
                         media_path: Optional[str] = None) -> int:
        """Get the number of times this question has failed."""
        key = self._generate_key(question_id, model_id, prompt, media_path)
        return self.failures.get(key, {}).get("count", 0)

    def clear_failures(self, question_id: Optional[str] = None) -> None:
        """Clear failure records. If question_id given, clear only that question's failures."""
        if question_id:
            self.failures = {k: v for k, v in self.failures.items()
                            if v.get("question_id") != question_id}
        else:
            self.failures.clear()
        self._save_cache()

    def store_consistency_responses(self, 
                                   question_id: str,
                                   model_id: str,
                                   prompt: str,
                                   responses: List[str],
                                   media_path: Optional[str] = None,
                                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store multiple responses for consistency checking"""
        key = f"{question_id}:{model_id}:consistency"
        
        entries = []
        for i, response in enumerate(responses):
            entry = CacheEntry(
                question_id=question_id,
                model_id=model_id,
                prompt=prompt,
                media_path=media_path,
                response=response,
                timestamp=datetime.now().isoformat(),
                metadata={**(metadata or {}), "consistency_run": i + 1}
            )
            entries.append(entry)
        
        self.consistency_cache[key] = entries
        self._save_cache()
        logger.debug(f"Cached {len(responses)} consistency responses for question {question_id}")
    
    def get_consistency_responses(self, 
                                 question_id: str,
                                 model_id: str) -> Optional[List[str]]:
        """Get cached consistency responses"""
        key = f"{question_id}:{model_id}:consistency"
        entries = self.consistency_cache.get(key)
        
        if entries:
            logger.debug(f"Consistency cache hit for question {question_id}")
            return [entry.response for entry in entries]
        
        return None
    
    def check_consistency(self, 
                         question_id: str,
                         model_id: str,
                         threshold: float = 0.8) -> Dict[str, Any]:
        """Check consistency of cached responses"""
        key = f"{question_id}:{model_id}:consistency"
        entries = self.consistency_cache.get(key)
        
        if not entries or len(entries) < 2:
            return {
                "is_consistent": True,
                "consistency_score": 1.0,
                "num_responses": len(entries) if entries else 0,
                "details": "Insufficient responses for consistency check"
            }
        
        responses = [entry.response for entry in entries]
        
        # Calculate pairwise similarity
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = self._calculate_similarity(responses[i], responses[j])
                similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        is_consistent = avg_similarity >= threshold
        
        return {
            "is_consistent": is_consistent,
            "consistency_score": avg_similarity,
            "num_responses": len(responses),
            "similarities": similarities,
            "threshold": threshold,
            "details": f"Average similarity: {avg_similarity:.3f}"
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity between two responses"""
        # Simple word-based Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_responses = len(self.cache)
        consistency_groups = len(self.consistency_cache)
        
        # Count by model
        model_counts = {}
        for entry in self.cache.values():
            model_counts[entry.model_id] = model_counts.get(entry.model_id, 0) + 1
        
        # Count by question
        question_counts = {}
        for entry in self.cache.values():
            question_counts[entry.question_id] = question_counts.get(entry.question_id, 0) + 1
        
        return {
            "total_responses": total_responses,
            "consistency_groups": consistency_groups,
            "unique_models": len(model_counts),
            "unique_questions": len(question_counts),
            "model_counts": model_counts,
            "question_counts": question_counts
        }
    
    def clear_cache(self, model_id: Optional[str] = None) -> None:
        """Clear cache, optionally for specific model"""
        if model_id:
            # Clear specific model
            keys_to_remove = [
                key for key, entry in self.cache.items()
                if entry.model_id == model_id
            ]
            for key in keys_to_remove:
                del self.cache[key]
            
            # Clear consistency cache for model
            keys_to_remove = [
                key for key in self.consistency_cache.keys()
                if key.endswith(f":{model_id}:consistency")
            ]
            for key in keys_to_remove:
                del self.consistency_cache[key]
        else:
            # Clear all
            self.cache.clear()
            self.consistency_cache.clear()
        
        self._save_cache()
        logger.info(f"Cleared cache for model: {model_id or 'all'}")
    
    def export_cache(self, output_file: Union[str, os.PathLike]) -> None:
        """Export cache to file"""
        output_path = output_file
        
        export_data = {
            "responses": {key: entry.to_dict() for key, entry in self.cache.items()},
            "consistency": {
                key: [entry.to_dict() for entry in entries]
                for key, entries in self.consistency_cache.items()
            },
            "stats": self.get_cache_stats(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported cache to {output_path}")
    
    def import_cache(self, input_file: Union[str, os.PathLike]) -> None:
        """Import cache from file"""
        input_path = input_file
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Cache file not found: {input_path}")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Import responses
        if "responses" in data:
            for key, entry_data in data["responses"].items():
                self.cache[key] = CacheEntry.from_dict(entry_data)
        
        # Import consistency cache
        if "consistency" in data:
            for key, entries_data in data["consistency"].items():
                self.consistency_cache[key] = [
                    CacheEntry.from_dict(entry_data)
                    for entry_data in entries_data
                ]
        
        self._save_cache()
        logger.info(f"Imported cache from {input_path}")


if __name__ == "__main__":
    cache = ResultCache()
    cache.clear_cache()

