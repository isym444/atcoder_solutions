#!/usr/bin/env python3
# from typing import *

YES = 'Yes'
NO = 'No'


# def solve(N: int, Q: int, T: List[int], A: List[int], B: List[int]) -> Any:
def solve(N, Q, T, A, B):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, Q = map(int, input().split())
    T = [None for _ in range(Q)]
    A = [None for _ in range(Q)]
    B = [None for _ in range(Q)]
    for i in range(Q):
        T[i], A[i], B[i] = map(int, input().split())
    ans = solve(N, Q, T, A, B)
    print(ans)  # TODO: edit here


if __name__ == '__main__':
    main()
