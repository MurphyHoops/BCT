def is_prime(n: int) -> bool:
    """Buggy prime checker (intentionally incorrect for MVP)."""
    if n < 2:
        return False
    # BUG: incorrectly labels 9 as prime (and other composites)
    for d in range(2, n - 1):
        if n % d == 0:
            return False
    return True
