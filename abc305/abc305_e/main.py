#!/usr/bin/env python3
# from typing import *


# def solve(N: str, M: str, K: str, a: List[str], b: List[str], p: List[str], h: List[str]) -> Tuple[int, List[int]]:
def solve(N, M, K, a, b, p, h):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, M, K = input().split()
    a = [None for _ in range(M)]
    b = [None for _ in range(M)]
    p = [None for _ in range(K)]
    h = [None for _ in range(K)]
    for i in range(M):
        a[i], b[i] = input().split()
    for i in range(K):
        p[i], h[i] = input().split()
    G, v = solve(N, M, K, a, b, p, h)
    print(G)
    print(*[v[i] for i in range(G)])


if __name__ == '__main__':
    main()
