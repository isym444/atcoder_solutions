#!/usr/bin/env python3
# from typing import *


# def solve(N: int, M: int, S: List[str]) -> Tuple[List[str], str]:
def solve(N, M, S):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, M = map(int, input().split())
    S = [None for _ in range(N)]
    for i in range(N):
        S[i] = input()
    c, d = solve(N, M, S)
    print(*[c[i] for i in range(M)], d)


if __name__ == '__main__':
    main()
