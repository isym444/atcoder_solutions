#!/usr/bin/env python3
# from typing import *



# def solve(N: int, M: int, X: List[int], A: List[int]) -> int:
def solve(N, M, X, A):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    N = int(next(tokens))
    M = int(next(tokens))
    X = [None for _ in range(M)]
    A = [None for _ in range(M)]
    for i in range(M):
        X[i] = int(next(tokens))
    for i in range(M):
        A[i] = int(next(tokens))
    assert next(tokens, None) is None
    a = solve(N, M, X, A)
    print(a)

if __name__ == '__main__':
    main()
