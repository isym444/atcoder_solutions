#!/usr/bin/env python3
# from typing import *



# def solve(N: int, X: List[int], Y: List[int]) -> Tuple[str, List[str], List[str]]:
def solve(N, X, Y):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N = int(input())
    X = [None for _ in range(N)]
    Y = [None for _ in range(N)]
    for i in range(N):
        X[i], Y[i] = map(int, input().split())
    m, d, w = solve(N, X, Y)
    print(m)
    print(*[d[i] for i in range(m)])
    for i in range(N):
        print(w[i])

if __name__ == '__main__':
    main()
