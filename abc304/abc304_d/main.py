#!/usr/bin/env python3
# from typing import *



# def solve(W: int, H: int, N: int, p: List[int], q: List[int], A: int, a: List[int], B: int, b: List[int]) -> Tuple[int, int]:
def solve(W, H, N, p, q, A, a, B, b):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    W = int(next(tokens))
    H = int(next(tokens))
    N = int(next(tokens))
    p = [None for _ in range(N)]
    q = [None for _ in range(N)]
    for i in range(N):
        p[i] = int(next(tokens))
        q[i] = int(next(tokens))
    A = int(next(tokens))
    a = [None for _ in range(A)]
    for i in range(A):
        a[i] = int(next(tokens))
    B = int(next(tokens))
    b = [None for _ in range(B)]
    for i in range(B):
        b[i] = int(next(tokens))
    assert next(tokens, None) is None
    m, M = solve(W, H, N, p, q, A, a, B, b)
    print(m, M)

if __name__ == '__main__':
    main()
