#!/usr/bin/env python3
# from typing import *



# def solve(N: int, Q: int, S: str, l: List[int], r: List[int]) -> List[str]:
def solve(N, Q, S, l, r):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    N, Q = map(int, input().split())
    l = [None for _ in range(Q)]
    r = [None for _ in range(Q)]
    S = input()
    for i in range(Q):
        l[i], r[i] = map(int, input().split())
    a = solve(N, Q, S, l, r)
    for i in range(Q):
        print(a[i])

if __name__ == '__main__':
    main()
