#!/usr/bin/env python3
# from typing import *



# def solve(N: int, l: List[int], r: List[int]) -> int:
def solve(N, l, r):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N = int(input())
    l = [None for _ in range(N)]
    r = [None for _ in range(N)]
    for i in range(N):
        l[i], r[i] = map(int, input().split())
    a = solve(N, l, r)
    print(a)

if __name__ == '__main__':
    main()