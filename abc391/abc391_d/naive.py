#!/usr/bin/env python3
# from typing import *

YES = 'Yes'
NO = 'No'

# def solve(N: int, W: int, X: List[int], Y: List[int], Q: int, T: List[int], A: List[int]) -> List[str]:
def solve(N, W, X, Y, Q, T, A):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, W = map(int, input().split())
    X = [None for _ in range(N)]
    Y = [None for _ in range(N)]
    for i in range(N):
        X[i], Y[i] = map(int, input().split())
    Q = int(input())
    T = [None for _ in range(Q)]
    A = [None for _ in range(Q)]
    for i in range(Q):
        T[i], A[i] = map(int, input().split())
    a = solve(N, W, X, Y, Q, T, A)
    for i in range(Q):
        print(a[i])

if __name__ == '__main__':
    main()
