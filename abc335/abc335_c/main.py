#!/usr/bin/env python3
# from typing import *



# def solve(a: str, b: str, c: str, d: str, e: List[str], f: List[str]) -> Tuple[str, List[str], List[str], str]:
def solve(a, b, c, d, e, f):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    a, b = input().split()
    e = [None for _ in range(b)]
    f = [None for _ in range(b)]
    c, d = input().split()
    for i in range(b):
        e[i], f[i] = input().split()
    a, b, c, d = solve(a, b, c, d, e, f)
    print(a, end=' ')
    for i in range(a):
        print(b[i])
        print(c[i], end=' ')
    print(d)

if __name__ == '__main__':
    main()
