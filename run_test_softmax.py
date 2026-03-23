import random

from fused_softmax import test_softmax

NUM_CASES = 10
MAX_DIM = 4096
SEED = 20260323


def generate_random_sizes(num_cases: int = NUM_CASES, max_dim: int = MAX_DIM) -> list[tuple[int, int]]:
    random.seed(SEED)
    return [
        (random.randint(1, max_dim), random.randint(1, max_dim))
        for _ in range(num_cases)
    ]


if __name__ == "__main__":
    for size in generate_random_sizes():
        test_softmax(size=size)
