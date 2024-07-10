import random

def generate_random_inputs():
    # Constraints
    N = random.randint(1, 2 * 10)
    M = random.randint(1, 2 * 10)
    
    # Ensure T_i values are strictly increasing
    T = sorted(random.sample(range(1, 100 + 1), M))
    W = [random.randint(1, 100) for _ in range(M)]
    S = [random.randint(1, 100) for _ in range(M)]
    
    return N, M, T, W, S

def main():
    N, M, T, W, S = generate_random_inputs()
    
    # Output the results in the specified format
    print(N, M)
    for i in range(M):
        print(T[i], W[i], S[i])

if __name__ == "__main__":
    main()
