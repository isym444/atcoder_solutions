#!/usr/bin/env python3
# from typing import *


# def solve(N: int, K: int, t: List[int], y: List[int]) -> int:
def solve(N, K, t, y):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, K = map(int, input().split())
    t = [None for _ in range(N)]
    y = [None for _ in range(N)]
    for i in range(N):
        t[i], y[i] = map(int, input().split())
    a = solve(N, K, t, y)
    print(a)


if __name__ == '__main__':
    main()
