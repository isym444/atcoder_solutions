#!/usr/bin/env python3
# from typing import *


# def solve(N: int, P: int, Q: int, R: int, S: int, A: List[int]) -> List[str]:
def solve(N, P, Q, R, S, A):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    N = int(next(tokens))
    P = int(next(tokens))
    Q = int(next(tokens))
    R = int(next(tokens))
    S = int(next(tokens))
    A = [None for _ in range(N)]
    for i in range(N):
        A[i] = int(next(tokens))
    assert next(tokens, None) is None
    ans = solve(N, P, Q, R, S, A)
    print(*[ans[i] for i in range(N)])


if __name__ == '__main__':
    main()
