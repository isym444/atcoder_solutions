#!/usr/bin/env python3
# from typing import *

MOD = 998244353


# def solve(N: int, M: int, L: List[int], R: List[int]) -> int:
def solve(N, M, L, R):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, M = map(int, input().split())
    L = [None for _ in range(M)]
    R = [None for _ in range(M)]
    for i in range(M):
        L[i], R[i] = map(int, input().split())
    a = solve(N, M, L, R)
    print(a)


if __name__ == '__main__':
    main()