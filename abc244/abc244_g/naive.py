#!/usr/bin/env python3
# from typing import *



# def solve(N: int, M: int, u: List[int], v: List[int], S: str) -> Tuple[str, List[str]]:
def solve(N, M, u, v, S):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, M = map(int, input().split())
    u = [None for _ in range(M)]
    v = [None for _ in range(M)]
    for i in range(M):
        u[i], v[i] = map(int, input().split())
    S = input()
    K, A = solve(N, M, u, v, S)
    print(K)
    print(*[A[i] for i in range(K)])

if __name__ == '__main__':
    main()
