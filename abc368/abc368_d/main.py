#!/usr/bin/env python3
# from typing import *



# def solve(N: int, K: int, A: List[int], B: List[int], V: List[int]) -> int:
def solve(N, K, A, B, V):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    N = int(next(tokens))
    K = int(next(tokens))
    A = [None for _ in range(N - 1)]
    B = [None for _ in range(N - 1)]
    V = [None for _ in range(K)]
    for i in range(N - 1):
        A[i] = int(next(tokens))
        B[i] = int(next(tokens))
    for i in range(K):
        V[i] = int(next(tokens))
    assert next(tokens, None) is None
    a = solve(N, K, A, B, V)
    print(a)

if __name__ == '__main__':
    main()