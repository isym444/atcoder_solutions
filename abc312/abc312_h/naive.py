#!/usr/bin/env python3
# from typing import *


# def solve(N: int, S: List[str]) -> List[str]:
def solve(N, S):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N = int(input())
    S = [None for _ in range(N)]
    for i in range(N):
        S[i] = input()
    ans = solve(N, S)
    print(*[ans[i] for i in range(N)])


if __name__ == '__main__':
    main()
