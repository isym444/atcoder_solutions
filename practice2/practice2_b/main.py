#!/usr/bin/env python3
# from typing import *



# def solve(a: str, b: str, c: str, d: str, e: List[str], f: List[str], g: List[str]) -> Tuple[int, int, int, int]:
def solve(a, b, c, d, e, f, g):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    a = next(tokens)
    b = next(tokens)
    e = [None for _ in range(b)]
    f = [None for _ in range(b)]
    g = [None for _ in range(b)]
    c = next(tokens)
    d = next(tokens)
    for i in range(b):
        e[i] = next(tokens)
        f[i] = next(tokens)
        g[i] = next(tokens)
    assert next(tokens, None) is None
    a, b, c, d = solve(a, b, c, d, e, f, g)
    print(a)
    print(b)
    print(c)
    print(d)

if __name__ == '__main__':
    main()
