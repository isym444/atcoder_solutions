#!/usr/bin/env python3
# from typing import *



# def solve(N: str, M: str, L: str, a: List[str], b: List[str], c: List[str], d: List[str]) -> int:
def solve(N, M, L, a, b, c, d):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    N = next(tokens)
    M = next(tokens)
    L = next(tokens)
    a = [None for _ in range(N)]
    b = [None for _ in range(M)]
    c = [None for _ in range(L)]
    d = [None for _ in range(L)]
    for i in range(N):
        a[i] = next(tokens)
    for i in range(M):
        b[i] = next(tokens)
    for i in range(L):
        c[i] = next(tokens)
        d[i] = next(tokens)
    assert next(tokens, None) is None
    a1 = solve(N, M, L, a, b, c, d)
    print(a1)

if __name__ == '__main__':
    main()
