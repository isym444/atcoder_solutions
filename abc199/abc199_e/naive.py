#!/usr/bin/env python3
# from typing import *



# def solve(N: str, M: str, X: List[str], Y: List[str], Z: List[str]) -> int:
def solve(N, M, X, Y, Z):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, M = input().split()
    X = [None for _ in range(M)]
    Y = [None for _ in range(M)]
    Z = [None for _ in range(M)]
    for i in range(M):
        X[i], Y[i], Z[i] = input().split()
    a = solve(N, M, X, Y, Z)
    print(a)

if __name__ == '__main__':
    main()
