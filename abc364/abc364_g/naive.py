#!/usr/bin/env python3
# from typing import *



# def solve(N: int, M: int, K: int, A: List[int], B: List[int], C: List[int]) -> Any:
def solve(N, M, K, A, B, C):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, M, K = map(int, input().split())
    A = [None for _ in range(M)]
    B = [None for _ in range(M)]
    C = [None for _ in range(M)]
    for i in range(M):
        A[i], B[i], C[i] = map(int, input().split())
    ans = solve(N, M, K, A, B, C)
    print(ans)  # TODO: edit here

if __name__ == '__main__':
    main()