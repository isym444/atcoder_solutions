#!/usr/bin/env python3
# from typing import *



# def solve(a: int, b: List[int], c: List[List[int]], d: List[List[int]]) -> int:
def solve(a, b, c, d):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    a = int(input())
    b = [None for _ in range(a)]
    c = [[None for _ in range(b_i)] for _ in range(a)]
    d = [[None for _ in range(b_i)] for _ in range(a)]
    for i in range(a):
        b[i] = int(input())
        for j in range(b_i):
            c[i][j], d[i][j] = map(int, input().split())
    a1 = solve(a, b, c, d)
    print(a1)

if __name__ == '__main__':
    main()
