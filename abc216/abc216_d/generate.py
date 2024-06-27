import random

def generate_test_case():
    # Constraints
    mmm = 200000
    N = random.randint(1, mmm)
    M = random.randint(2, mmm)

    # The sum of all ki should be 2N
    k = []
    total = 2 * N
    for _ in range(M - 1):
        val = random.randint(1, total - (M - len(k) - 1))
        k.append(val)
        total -= val
    k.append(total)
    random.shuffle(k)

    # Generate the ai,j values
    values = list(range(1, N + 1)) * 2
    random.shuffle(values)
    idx = 0

    output = f"{N} {M}\n"
    for i in range(M):
        output += f"{k[i]}\n"
        output += " ".join(str(values[idx + j]) for j in range(k[i])) + "\n"
        idx += k[i]

    return output

# Generate and print a test case
print(generate_test_case())
