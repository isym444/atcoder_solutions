#!/usr/bin/env python3
# from typing import *



# def solve(N: int, K: int, X: List[int], A: List[int]) -> List[str]:
def solve(N, K, X, A):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    N = int(next(tokens))
    K = int(next(tokens))
    X = [None for _ in range(N)]
    A = [None for _ in range(N)]
    for i in range(N):
        X[i] = int(next(tokens))
    for i in range(N):
        A[i] = int(next(tokens))
    assert next(tokens, None) is None
    ans = solve(N, K, X, A)
    print(*[ans[i] for i in range(N)])

if __name__ == '__main__':
    main()
