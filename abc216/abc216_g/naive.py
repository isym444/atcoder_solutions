#!/usr/bin/env python3
# from typing import *



# def solve(N: int, M: int, L: List[int], R: List[int], X: List[int]) -> List[str]:
def solve(N, M, L, R, X):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, M = map(int, input().split())
    L = [None for _ in range(M)]
    R = [None for _ in range(M)]
    X = [None for _ in range(M)]
    for i in range(M):
        L[i], R[i], X[i] = map(int, input().split())
    A = solve(N, M, L, R, X)
    print(*[A[i] for i in range(N)])

if __name__ == '__main__':
    main()
