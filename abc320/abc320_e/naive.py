#!/usr/bin/env python3
# from typing import *



# def solve(N: int, M: int, T: List[int], W: List[int], S: List[int]) -> List[str]:
def solve(N, M, T, W, S):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, M = map(int, input().split())
    T = [None for _ in range(M)]
    W = [None for _ in range(M)]
    S = [None for _ in range(M)]
    for i in range(M):
        T[i], W[i], S[i] = map(int, input().split())
    ans = solve(N, M, T, W, S)
    for i in range(N):
        print(ans[i])

if __name__ == '__main__':
    main()
