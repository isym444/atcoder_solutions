#!/usr/bin/env python3
# from typing import *


# def solve(N: int, A: List[int], B: List[int], U: List[int], V: List[int]) -> List[str]:
def solve(N, A, B, U, V):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N = int(input())
    A = [None for _ in range(N)]
    B = [None for _ in range(N)]
    U = [None for _ in range(N - 1)]
    V = [None for _ in range(N - 1)]
    for i in range(N):
        A[i], B[i] = map(int, input().split())
    for i in range(N - 1):
        U[i], V[i] = map(int, input().split())
    a = solve(N, A, B, U, V)
    print(*[a[i] for i in range(N)])


if __name__ == '__main__':
    main()
