#!/usr/bin/env python3
# from typing import *

YES = 'Yes'
NO = 'No'


# def solve(H: str, W: str, A: List[List[str]], N: str, R: List[str], C: List[str], E: List[str]) -> str:
def solve(H, W, A, N, R, C, E):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    H = next(tokens)
    W = next(tokens)
    A = [[None for _ in range(H + W + 4)] for _ in range(H + W + 4)]
    for j in range(H + 4):
        for i in range(W):
            A[i + j][i + j] = next(tokens)
    N = next(tokens)
    R = [None for _ in range(N)]
    C = [None for _ in range(N)]
    E = [None for _ in range(N)]
    for i in range(N):
        R[i] = next(tokens)
        C[i] = next(tokens)
        E[i] = next(tokens)
    assert next(tokens, None) is None
    a = solve(H, W, A, N, R, C, E)
    print(a)


if __name__ == '__main__':
    main()
