#!/usr/bin/env python3
# from typing import *

MOD = 998244353

# def solve(N: int, c: List[int]) -> List[str]:
def solve(N, c):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    N = int(next(tokens))
    c = [None for _ in range(N)]
    for i in range(N):
        c[i] = int(next(tokens))
    assert next(tokens, None) is None
    ans = solve(N, c)
    for i in range(N):
        print(ans[i])

if __name__ == '__main__':
    main()
