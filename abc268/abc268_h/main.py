#!/usr/bin/env python3
# from typing import *


# def solve(S: str, N: int, T: List[str]) -> int:
def solve(S, N, T):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    S = input()
    N = int(input())
    T = [None for _ in range(N)]
    for i in range(N):
        T[i] = input()
    a = solve(S, N, T)
    print(a)


if __name__ == '__main__':
    main()