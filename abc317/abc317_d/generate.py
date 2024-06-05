import random

# Define the constraints
N = random.randint(1,6)  # 1 ≤ N ≤ 100
X = []
Y = []
Z = []

# Generate values for X, Y, Z based on the constraints
for _ in range(N):
    x = random.randint(0, 100)
    while True:
        y = random.randint(0, 1e9)
        if (x + y) % 2 == 1:  # Ensure X_i + Y_i is odd
            break
    z = random.randint(1, 1e5)

    X.append(x)
    Y.append(y)
    Z.append(z)

# Ensure the sum of Z is odd
sum_Z = sum(Z)
if sum_Z % 2 == 0:
    Z[-1] += 1  # Adjust the last element to make the sum odd

# Ensure the sum of Z does not exceed 10^5
while sum(Z) > 10**5:
    Z[-1] -= 1

# Print the values in the specified format
print(N)
for i in range(N):
    print(X[i], Y[i], Z[i])
