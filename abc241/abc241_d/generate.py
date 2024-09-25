import random

def generate_random_input():
    # Set constraints
    Q = random.randint(1, 4)  # Number of queries, between 1 and 200,000
    max_x = 20  # Maximum value for x
    max_k = 5  # Maximum value for k
    
    # Output the value of Q
    print(Q)
    
    for _ in range(Q):
        query_type = random.randint(1, 3)  # Randomly choose query type 1, 2, or 3
        x = random.randint(1, max_x)  # Random x value in range [1, 10^18]
        
        if query_type == 1:
            # Query type 1: "1 x"
            print(f"1 {x}")
        else:
            # Query type 2 or 3: "2 x k" or "3 x k"
            k = random.randint(1, max_k)  # Random k value in range [1, 5]
            print(f"{query_type} {x} {k}")

# Run the function to generate random input
generate_random_input()