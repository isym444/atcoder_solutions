#!/usr/bin/env python3
# from typing import *



# def solve(N: int, R: int, D: List[int], A: List[int]) -> int:
def solve(N, R, D, A):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, R = map(int, input().split())
    D = [None for _ in range(N)]
    A = [None for _ in range(N)]
    for i in range(N):
        D[i], A[i] = map(int, input().split())
    a = solve(N, R, D, A)
    print(a)

if __name__ == '__main__':
    main()
