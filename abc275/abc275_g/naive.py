#!/usr/bin/env python3
# from typing import *


# def solve(N: int, A: List[int], B: List[int], C: List[int]) -> float:
def solve(N, A, B, C):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N = int(input())
    A = [None for _ in range(N)]
    B = [None for _ in range(N)]
    C = [None for _ in range(N)]
    for i in range(N):
        A[i], B[i], C[i] = map(int, input().split())
    a = solve(N, A, B, C)
    print(a)


if __name__ == '__main__':
    main()
