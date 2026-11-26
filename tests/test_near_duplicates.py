import random

from judge_audit.near_duplicates import jaccard, shingles, similar_pairs


def test_shingles_ignore_case_and_punctuation():
    assert shingles("One two, THREE four five!") == shingles("one two three four five")
    assert shingles("short prompt") == frozenset(["short prompt"])
    assert shingles("   ") == frozenset()


def test_similar_pairs_matches_brute_force():
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(30)]
    base = [[rng.choice(vocab) for _ in range(rng.randint(3, 25))] for _ in range(40)]
    # Add lightly edited copies so some pairs land near the threshold.
    docs = base + [words[:-1] + ["edit"] for words in base[:20]]
    sets = [shingles(" ".join(words)) for words in docs]
    left, right = sets[:30], sets[30:]
    for threshold in (0.5, 0.8, 1.0):
        expected = {(i, j) for i, a in enumerate(left) for j, b in enumerate(right) if a and jaccard(a, b) >= threshold}
        assert {(i, j) for i, j, _ in similar_pairs(left, right, threshold)} == expected
