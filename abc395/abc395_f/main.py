#!/usr/bin/env python3
# from typing import *



# def solve(N: int, X: int, U: List[int], D: List[int]) -> int:
def solve(N, X, U, D):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, X = map(int, input().split())
    U = [None for _ in range(N)]
    D = [None for _ in range(N)]
    for i in range(N):
        U[i], D[i] = map(int, input().split())
    a = solve(N, X, U, D)
    print(a)

if __name__ == '__main__':
    main()
