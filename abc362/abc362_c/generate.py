import random

def generate_input(N, L_min, L_max, R_min, R_max):
    inputs = []
    inputs.append(str(N))
    
    for _ in range(N):
        L=2
        R=0
        while(L>R):
            L = random.randint(L_min, L_max)
            R = random.randint(R_min, R_max)
        inputs.append(f"{L} {R}")
    
    return "\n".join(inputs)

# Parameters
N = 10  # Number of intervals
L_min, L_max = -20, 20
R_min, R_max = -20, 20

# Generate and print input
generated_input = generate_input(N, L_min, L_max, R_min, R_max)
print(generated_input)
