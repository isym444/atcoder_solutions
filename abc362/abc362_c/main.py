#!/usr/bin/env python3
# from typing import *

YES = 'Yes'
NO = 'No'

# def solve(N: int, L: List[int], R: List[int]) -> Tuple[str, List[str]]:
def solve(N, L, R):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N = int(input())
    L = [None for _ in range(N)]
    R = [None for _ in range(N)]
    for i in range(N):
        L[i], R[i] = map(int, input().split())
    Yes, X = solve(N, L, R)
    print(Yes)
    print(*[X[i] for i in range(N)])

if __name__ == '__main__':
    main()
