#!/usr/bin/env python3
# from typing import *


# def solve(N: int, M: int, p: List[int], x: List[int], y: List[int]) -> int:
def solve(N, M, p, x, y):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    N = int(next(tokens))
    M = int(next(tokens))
    p = [None for _ in range(N - 1)]
    x = [None for _ in range(M)]
    y = [None for _ in range(M)]
    for i in range(N - 1):
        p[i] = int(next(tokens))
    for i in range(M):
        x[i] = int(next(tokens))
        y[i] = int(next(tokens))
    assert next(tokens, None) is None
    a = solve(N, M, p, x, y)
    print(a)


if __name__ == '__main__':
    main()
