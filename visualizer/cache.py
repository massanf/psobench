import pathlib
import hashlib
import pickle
from typing import Any
from constants import CACHE, DATA


def compute_attempt_checksum(attempt_path: pathlib.Path) -> str:
    """Compute a lightweight checksum based on file metadata for an attempt."""
    hash_sha256 = hashlib.sha256()
    for file_path in sorted(attempt_path.iterdir()):
        if file_path.is_file():
            stat = file_path.stat()
            # Incorporate file name, size, and modification time
            hash_sha256.update(file_path.name.encode())
            hash_sha256.update(str(stat.st_size).encode())
            hash_sha256.update(str(stat.st_mtime).encode())
    return hash_sha256.hexdigest()


def load_attempt_cache(
    problem: str, attempt: str, analysis_content: str
) -> Any:
    """Load cached result and checksum for a given problem and attempt."""
    cache_file = CACHE / problem / f"{attempt}_{analysis_content}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None


def save_attempt_cache(
    problem: str,
    attempt: str,
    checksum: str,
    result: Any,
    analysis_content: str,
) -> None:
    """Save result and checksum to cache for a given problem and attempt."""
    problem_cache_dir = CACHE / problem
    problem_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = problem_cache_dir / f"{attempt}_{analysis_content}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump((checksum, result), f)


def process_attempt_cached(args: Any) -> Any:
    """
    Wrapper function to handle caching for a single attempt.
    'args' is expected to be a tuple: (attempt, problem, analysis_content)
    """
    attempt, problem, analysis_content = args
    attempt_path = DATA / problem / attempt
    checksum = compute_attempt_checksum(attempt_path)

    cached = load_attempt_cache(problem, attempt, analysis_content)
    if cached is not None:
        cached_checksum, cached_result = cached
    else:
        cached_checksum, cached_result = None, None

    if cached_checksum == checksum and cached_result is not None:
        return cached_result
    else:
        try:
            import rmt
            result = rmt.process_attempt(args)
            save_attempt_cache(
                problem, attempt, checksum, result, analysis_content
            )
            return result
        except Exception as e:
            print(f"Error on {problem} {attempt}: {e}")
            return None
