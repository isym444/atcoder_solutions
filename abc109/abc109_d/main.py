#!/usr/bin/env python3
# from typing import *


# def solve(a: str, b: str, c: List[str], d: List[List[str]]) -> Tuple[int, List[int], List[int], List[int], List[int]]:
def solve(a, b, c, d):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    a = next(tokens)
    b = next(tokens)
    c = [None for _ in range(a)]
    d = [[None for _ in range(a)] for _ in range(a)]
    for i in range(a):
        c[i] = next(tokens)
        for j in range(i):
            d[i][j] = next(tokens)
    assert next(tokens, None) is None
    a, b, c, d, e = solve(a, b, c, d)
    print(a)
    for i in range(a):
        print(b[i], c[i], d[i], e[i])


if __name__ == '__main__':
    main()
