#!/usr/bin/env python3
# from typing import *



# def solve(a: int, b: List[int], c: List[int], d: List[int], e: List[int], f: List[int], g: List[int]) -> List[str]:
def solve(a, b, c, d, e, f, g):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    a = int(input())
    b = [None for _ in range(a)]
    c = [None for _ in range(a)]
    d = [None for _ in range(a)]
    e = [None for _ in range(a)]
    f = [None for _ in range(a)]
    g = [None for _ in range(a)]
    for i in range(a):
        b[i], c[i], d[i], e[i], f[i], g[i] = map(int, input().split())
    h = solve(a, b, c, d, e, f, g)
    for j in range(a):
        print(h[j])

if __name__ == '__main__':
    main()
