#!/usr/bin/env python3
# from typing import *

YES = 'Yes'
NO = 'No'


# def solve(N: int, a: List[int], b: List[int], Q: int, x: List[int], y: List[int]) -> List[str]:
def solve(N, a, b, Q, x, y):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    N = int(next(tokens))
    a = [None for _ in range(N)]
    b = [None for _ in range(N)]
    for i in range(N):
        a[i] = int(next(tokens))
    for i in range(N):
        b[i] = int(next(tokens))
    Q = int(next(tokens))
    x = [None for _ in range(Q)]
    y = [None for _ in range(Q)]
    for i in range(Q):
        x[i] = int(next(tokens))
        y[i] = int(next(tokens))
    assert next(tokens, None) is None
    c = solve(N, a, b, Q, x, y)
    for i in range(Q):
        print(c[i])


if __name__ == '__main__':
    main()
