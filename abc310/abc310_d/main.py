#!/usr/bin/env python3
# from typing import *



# def solve(N: str, T: str, M: str, A: List[str], B: List[str]) -> int:
def solve(N, T, M, A, B):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, T, M = input().split()
    A = [None for _ in range(M)]
    B = [None for _ in range(M)]
    for i in range(M):
        A[i], B[i] = input().split()
    a = solve(N, T, M, A, B)
    print(a)

if __name__ == '__main__':
    main()
