#!/usr/bin/env python3
# from typing import *



# def solve(X: int, Y: int, A: int, B: int, C: int, p: List[int], q: List[int], r: List[int]) -> int:
def solve(X, Y, A, B, C, p, q, r):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    X = int(next(tokens))
    Y = int(next(tokens))
    A = int(next(tokens))
    B = int(next(tokens))
    C = int(next(tokens))
    p = [None for _ in range(A)]
    q = [None for _ in range(B)]
    r = [None for _ in range(C)]
    for i in range(A):
        p[i] = int(next(tokens))
    for i in range(B):
        q[i] = int(next(tokens))
    for i in range(C):
        r[i] = int(next(tokens))
    assert next(tokens, None) is None
    a = solve(X, Y, A, B, C, p, q, r)
    print(a)

if __name__ == '__main__':
    main()
