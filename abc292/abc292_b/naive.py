#!/usr/bin/env python3
# from typing import *

YES = 'Yes'
NO = 'No'

# def solve(a: int, b: int, c: List[int], d: List[int]) -> Tuple[List[str], List[str]]:
def solve(a, b, c, d):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    a, b = map(int, input().split())
    c = [None for _ in range(b)]
    d = [None for _ in range(b)]
    for i in range(b):
        c[i], d[i] = map(int, input().split())
    e, h = solve(a, b, c, d)
    for i in range(a):
        print(e[i])
        print(h[i])

if __name__ == '__main__':
    main()
