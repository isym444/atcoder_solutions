#!/usr/bin/env python3
# from typing import *

YES = 'Yes'
NO = 'No'

# def solve(N: int, M: int, Q: int, a: List[int], b: List[int], c: List[int], u: List[int], v: List[int], w: List[int]) -> List[str]:
def solve(N, M, Q, a, b, c, u, v, w):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, M, Q = map(int, input().split())
    a = [None for _ in range(M)]
    b = [None for _ in range(M)]
    c = [None for _ in range(M)]
    u = [None for _ in range(Q)]
    v = [None for _ in range(Q)]
    w = [None for _ in range(Q)]
    for i in range(M):
        a[i], b[i], c[i] = map(int, input().split())
    for i in range(Q):
        u[i], v[i], w[i] = map(int, input().split())
    d = solve(N, M, Q, a, b, c, u, v, w)
    for i in range(Q):
        print(d[i])

if __name__ == '__main__':
    main()
