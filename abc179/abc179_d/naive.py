#!/usr/bin/env python3
# from typing import *

MOD = 998244353

# def solve(N: int, K: int, L: List[int], R: List[int]) -> int:
def solve(N, K, L, R):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, K = map(int, input().split())
    L = [None for _ in range(K)]
    R = [None for _ in range(K)]
    for i in range(K):
        L[i], R[i] = map(int, input().split())
    a = solve(N, K, L, R)
    print(a)

if __name__ == '__main__':
    main()
