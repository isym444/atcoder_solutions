#!/usr/bin/env python3
# from typing import *

MOD = 998244353


# def solve(N: int, M: int, K: int, L: int, A: List[int], B: List[int]) -> int:
def solve(N, M, K, L, A, B):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    N = int(next(tokens))
    M = int(next(tokens))
    K = int(next(tokens))
    L = int(next(tokens))
    A = [None for _ in range(K)]
    B = [None for _ in range(L)]
    for i in range(K):
        A[i] = int(next(tokens))
    for i in range(L):
        B[i] = int(next(tokens))
    assert next(tokens, None) is None
    a = solve(N, M, K, L, A, B)
    print(a)


if __name__ == '__main__':
    main()
