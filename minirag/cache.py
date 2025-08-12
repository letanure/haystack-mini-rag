"""Simple file-based caching for embeddings."""

import hashlib
import os
import pickle
from pathlib import Path
from typing import List, Optional

from haystack import Document


class EmbeddingCache:
    """Simple disk cache for document embeddings."""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, docs_path: str, model_name: str) -> str:
        """Generate cache key from file path and model."""
        # Include file modification time to detect changes
        try:
            mtime = os.path.getmtime(docs_path)
            key_data = f"{docs_path}:{model_name}:{mtime}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except OSError:
            # File doesn't exist, return unique key
            return hashlib.md5(f"{docs_path}:{model_name}".encode()).hexdigest()

    def get_cached_docs(
        self, docs_path: str, model_name: str
    ) -> Optional[List[Document]]:
        """Load cached embedded documents if available."""
        cache_key = self._get_cache_key(docs_path, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, IOError):
            # Cache corrupted, remove it
            cache_file.unlink(missing_ok=True)
            return None

    def save_embedded_docs(
        self, docs_path: str, model_name: str, embedded_docs: List[Document]
    ) -> None:
        """Save embedded documents to cache."""
        cache_key = self._get_cache_key(docs_path, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(embedded_docs, f)
        except (pickle.PickleError, IOError) as e:
            print(f"Warning: Failed to save cache: {e}")

    def clear_cache(self) -> None:
        """Remove all cached files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink(missing_ok=True)

    def cache_info(self) -> dict:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "num_files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
        }
