import random

def generate_input(N, M):
    # Generate N random integers between 0 and M-1
    array = [random.randint(0, M-1) for _ in range(N)]
    return N, M, array

def print_input(N, M, array):
    print(f"{N} {M}")
    print(" ".join(map(str, array)))

# Example usage
N = random.randint(1,10)  # Number of elements
M = random.randint(2,10)  # Maximum value for elements
N, M, array = generate_input(N, M)
print_input(N, M, array)
