#!/usr/bin/env python3
# from typing import *



# def solve(N: int, X: int, Y: int, A: List[int], B: List[int]) -> int:
def solve(N, X, Y, A, B):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N = int(input())
    A = [None for _ in range(N)]
    B = [None for _ in range(N)]
    X, Y = map(int, input().split())
    for i in range(N):
        A[i], B[i] = map(int, input().split())
    a = solve(N, X, Y, A, B)
    print(a)

if __name__ == '__main__':
    main()
