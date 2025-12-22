import solution

def test_small_primes():
    assert solution.is_prime(2) is True
    assert solution.is_prime(3) is True
    assert solution.is_prime(5) is True
    assert solution.is_prime(7) is True

def test_small_composites():
    assert solution.is_prime(1) is False
    assert solution.is_prime(4) is False
    assert solution.is_prime(6) is False
    assert solution.is_prime(8) is False
    assert solution.is_prime(9) is False
    assert solution.is_prime(21) is False

def test_large_prime():
    assert solution.is_prime(9973) is True
