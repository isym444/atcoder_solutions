#!/usr/bin/env python3
# from typing import *


# def solve(N: int, T: int, t: List[int]) -> int:
def solve(N, T, t):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    N = int(next(tokens))
    T = int(next(tokens))
    t = [None for _ in range(N)]
    for i in range(N):
        t[i] = int(next(tokens))
    assert next(tokens, None) is None
    a = solve(N, T, t)
    print(a)


if __name__ == '__main__':
    main()
