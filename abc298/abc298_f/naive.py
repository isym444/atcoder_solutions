#!/usr/bin/env python3
# from typing import *


# def solve(N: int, r: List[int], c: List[int], x: List[int]) -> int:
def solve(N, r, c, x):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N = int(input())
    r = [None for _ in range(N)]
    c = [None for _ in range(N)]
    x = [None for _ in range(N)]
    for i in range(N):
        r[i], c[i], x[i] = map(int, input().split())
    a = solve(N, r, c, x)
    print(a)


if __name__ == '__main__':
    main()