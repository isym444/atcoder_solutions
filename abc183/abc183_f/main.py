#!/usr/bin/env python3
# from typing import *


# def solve(N: str, Q: str, C: List[str], Query: List[str]) -> Any:
def solve(N, Q, C, Query):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    N = next(tokens)
    Q = next(tokens)
    C = [None for _ in range(N)]
    Query = [None for _ in range(Q)]
    for i in range(N):
        C[i] = next(tokens)
    for i in range(Q):
        Query[i] = next(tokens)
    assert next(tokens, None) is None
    ans = solve(N, Q, C, Query)
    print(ans)  # TODO: edit here


if __name__ == '__main__':
    main()
