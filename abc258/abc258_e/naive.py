#!/usr/bin/env python3
# from typing import *


# def solve(N: int, Q: int, X: int, W: List[int], K: List[int]) -> List[str]:
def solve(N, Q, X, W, K):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    N = int(next(tokens))
    Q = int(next(tokens))
    X = int(next(tokens))
    W = [None for _ in range(N)]
    K = [None for _ in range(Q)]
    for i in range(N):
        W[i] = int(next(tokens))
    for i in range(Q):
        K[i] = int(next(tokens))
    assert next(tokens, None) is None
    a = solve(N, Q, X, W, K)
    for i in range(Q):
        print(a[i])


if __name__ == '__main__':
    main()
