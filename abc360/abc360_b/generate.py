import random
import string

def generate_input():
    S_length = random.randint(1, 20)
    T_length = random.randint(1, S_length)
    
    S = ''.join(random.choices(string.ascii_lowercase, k=S_length))
    T = ''.join(random.choices(string.ascii_lowercase, k=T_length))
    
    return S, T

# Generate a sample input
S, T = generate_input()
print(f"S: {S}")
print(f"T: {T}")
