#!/usr/bin/env python3
# from typing import *


# def solve(N: int, s: List[str]) -> List[str]:
def solve(N, s):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N = int(input())
    s = [None for _ in range(N)]
    for i in range(N):
        s[i] = input()
    ans = solve(N, s)
    for i in range(N):
        print(ans[i])


if __name__ == '__main__':
    main()
