#!/usr/bin/env python3
# from typing import *



# def solve(a: int, b: List[int], c: List[int], d: List[int], e: List[int], f: int, g: int) -> int:
def solve(a, b, c, d, e, f, g):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    a = int(next(tokens))
    b = [None for _ in range(a)]
    c = [None for _ in range(a)]
    d = [None for _ in range(a)]
    e = [None for _ in range(a)]
    for i in range(a):
        b[i] = int(next(tokens))
        c[i] = int(next(tokens))
        d[i] = int(next(tokens))
        e[i] = int(next(tokens))
    f = int(next(tokens))
    g = int(next(tokens))
    assert next(tokens, None) is None
    a1 = solve(a, b, c, d, e, f, g)
    print(a1)

if __name__ == '__main__':
    main()
