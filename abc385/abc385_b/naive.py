#!/usr/bin/env python3
# from typing import *



# def solve(H: str, W: str, X: str, Y: str, S: List[List[str]], T: str) -> Tuple[int, int, int]:
def solve(H, W, X, Y, S, T):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    H = next(tokens)
    W = next(tokens)
    X = next(tokens)
    Y = next(tokens)
    S = [[None for _ in range(W)] for _ in range(H + W + 2)]
    for j in range(H + 2):
        for i in range(W):
            S[i + j][i] = next(tokens)
    T = next(tokens)
    assert next(tokens, None) is None
    a, b, c = solve(H, W, X, Y, S, T)
    print(a, b, c)

if __name__ == '__main__':
    main()
