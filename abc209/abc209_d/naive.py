#!/usr/bin/env python3
# from typing import *


# def solve(N: int, Q: int, a: List[int], b: List[int], c: List[int], d: List[int]) -> List[str]:
def solve(N, Q, a, b, c, d):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, Q = map(int, input().split())
    a = [None for _ in range(N - 1)]
    b = [None for _ in range(N - 1)]
    c = [None for _ in range(Q)]
    d = [None for _ in range(Q)]
    for i in range(N - 1):
        a[i], b[i] = map(int, input().split())
    for i in range(Q):
        c[i], d[i] = map(int, input().split())
    e = solve(N, Q, a, b, c, d)
    for i in range(Q):
        print(e[i])


if __name__ == '__main__':
    main()
