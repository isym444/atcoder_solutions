#!/usr/bin/env python3
# from typing import *



# def solve(N: int, M: int, Q: int, A: List[int], B: List[int], S: List[int], T: List[int], X: List[int], Y: List[int], Z: List[int]) -> Any:
def solve(N, M, Q, A, B, S, T, X, Y, Z):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, M, Q = map(int, input().split())
    A = [None for _ in range(M)]
    B = [None for _ in range(M)]
    S = [None for _ in range(M)]
    T = [None for _ in range(M)]
    X = [None for _ in range(Q)]
    Y = [None for _ in range(Q)]
    Z = [None for _ in range(Q)]
    for i in range(M):
        A[i], B[i], S[i], T[i] = map(int, input().split())
    for i in range(Q):
        X[i], Y[i], Z[i] = map(int, input().split())
    ans = solve(N, M, Q, A, B, S, T, X, Y, Z)
    print(ans)  # TODO: edit here

if __name__ == '__main__':
    main()
