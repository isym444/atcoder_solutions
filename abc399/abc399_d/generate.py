import random

def generate_valid_input(num_cases=3, n_min=1, n_max=5):
    result = [str(num_cases)]
    for _ in range(num_cases):
        n = random.randint(n_min, n_max)
        # Generate list with two of each number from 0 to n-1
        values = [i for i in range(n) for _ in range(2)]
        random.shuffle(values)
        # Convert values to 1-based to match expected input format
        result.append(str(n))
        result.append(" ".join(str(v + 1) for v in values))
    return "\n".join(result)

if __name__ == "__main__":
    print(generate_valid_input())
