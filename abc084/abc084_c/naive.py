#!/usr/bin/env python3
# from typing import *



# def solve(N: int, C: List[int], S: List[int], F: List[int]) -> List[str]:
def solve(N, C, S, F):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N = int(input())
    C = [None for _ in range(N - 1)]
    S = [None for _ in range(N - 1)]
    F = [None for _ in range(N - 1)]
    for i in range(N - 1):
        C[i], S[i], F[i] = map(int, input().split())
    ans = solve(N, C, S, F)
    for i in range(N):
        print(ans[i])

if __name__ == '__main__':
    main()
