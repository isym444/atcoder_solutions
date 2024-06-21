A = [None for _ in range(N)]
    for i in range(N):
        A[i] = random.randint(1, 10)  # TODO: edit here
    print(N)
    print(*[A[i] for i in range(N)])