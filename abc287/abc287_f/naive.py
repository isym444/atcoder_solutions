#!/usr/bin/env python3
# from typing import *

MOD = 998244353


# def solve(N: int, a: List[int], b: List[int]) -> List[str]:
def solve(N, a, b):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N = int(input())
    a = [None for _ in range(N - 1)]
    b = [None for _ in range(N - 1)]
    for i in range(N - 1):
        a[i], b[i] = map(int, input().split())
    ans = solve(N, a, b)
    for i in range(N):
        print(ans[i])


if __name__ == '__main__':
    main()
