#!/usr/bin/env python3
# from typing import *


# def solve(N: int, K: int, C: List[int], V: List[int]) -> int:
def solve(N, K, C, V):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, K = map(int, input().split())
    C = [None for _ in range(N)]
    V = [None for _ in range(N)]
    for i in range(N):
        C[i], V[i] = map(int, input().split())
    a = solve(N, K, C, V)
    print(a)


if __name__ == '__main__':
    main()
