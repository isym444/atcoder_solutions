#!/usr/bin/env python3
# from typing import *



# def solve(N: int, M: int, A: List[int], B: List[int]) -> int:
def solve(N, M, A, B):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, M = map(int, input().split())
    A = [None for _ in range(M)]
    B = [None for _ in range(M)]
    for i in range(M):
        A[i], B[i] = map(int, input().split())
    b = solve(N, M, A, B)
    print(b)

if __name__ == '__main__':
    main()
