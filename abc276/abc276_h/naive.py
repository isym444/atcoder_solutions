#!/usr/bin/env python3
# from typing import *

YES = 'Yes'
NO = 'No'


# def solve(N: int, Q: int, a: List[int], b: List[int], c: List[int], d: List[int], e: List[int]) -> List[List[str]]:
def solve(N, Q, a, b, c, d, e):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, Q = map(int, input().split())
    a = [None for _ in range(Q)]
    b = [None for _ in range(Q)]
    c = [None for _ in range(Q)]
    d = [None for _ in range(Q)]
    e = [None for _ in range(Q)]
    for i in range(Q):
        a[i], b[i], c[i], d[i], e[i] = map(int, input().split())
    x = solve(N, Q, a, b, c, d, e)
    for j in range(N + 4):
        print(*[x[i + j][i + j] for i in range(N)])


if __name__ == '__main__':
    main()
