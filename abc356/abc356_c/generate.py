import random

def generate_input(N, M, K):
    C = [random.randint(1, N) for _ in range(M)]
    A = [[random.randint(1, N) for _ in range(C[i])] for i in range(M)]
    for i in range(M):
        # Ensure Ai,j != Ai,k if j != k
        while len(set(A[i])) != len(A[i]):
            A[i] = [random.randint(1, N) for _ in range(C[i])]
    
    # Ensure mixing 'o' and 'x'
    R = []
    num_o = num_x = 0
    for _ in range(M):
        if num_o > num_x:
            R.append('x')
            num_x += 1
        elif num_x > num_o:
            R.append('o')
            num_o += 1
        else:
            choice = random.choice(['o', 'x'])
            R.append(choice)
            if choice == 'o':
                num_o += 1
            else:
                num_x += 1

    # Print the generated input in the required format
    print(f"{N} {M} {K}")
    for i in range(M):
        print(f"{C[i]} {' '.join(map(str, A[i]))} {R[i]}")

# Example usage
N = random.randint(1, 5)
M = random.randint(1, 7)
K = random.randint(1, N)

generate_input(N, M, K)
