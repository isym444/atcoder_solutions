#!/usr/bin/env python3
# from typing import *


# def solve(N: int, M: int, K: int, u: List[int], v: List[int], a: List[int], s: List[int]) -> int:
def solve(N, M, K, u, v, a, s):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    N = int(next(tokens))
    M = int(next(tokens))
    K = int(next(tokens))
    u = [None for _ in range(M)]
    v = [None for _ in range(M)]
    a = [None for _ in range(M)]
    s = [None for _ in range(K)]
    for i in range(M):
        u[i] = int(next(tokens))
        v[i] = int(next(tokens))
        a[i] = int(next(tokens))
    for i in range(K):
        s[i] = int(next(tokens))
    assert next(tokens, None) is None
    a1 = solve(N, M, K, u, v, a, s)
    print(a1)


if __name__ == '__main__':
    main()
