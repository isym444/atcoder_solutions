#!/usr/bin/env python3
# from typing import *



# def solve(N: int, A: int, B: int, X: List[int]) -> int:
def solve(N, A, B, X):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    N = int(next(tokens))
    A = int(next(tokens))
    B = int(next(tokens))
    X = [None for _ in range(N)]
    for i in range(N):
        X[i] = int(next(tokens))
    assert next(tokens, None) is None
    a = solve(N, A, B, X)
    print(a)

if __name__ == '__main__':
    main()
