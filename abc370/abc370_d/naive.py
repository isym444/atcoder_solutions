#!/usr/bin/env python3
# from typing import *



# def solve(H: int, W: int, Q: int, R: List[int], C: List[int]) -> int:
def solve(H, W, Q, R, C):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    H, W, Q = map(int, input().split())
    R = [None for _ in range(Q)]
    C = [None for _ in range(Q)]
    for i in range(Q):
        R[i], C[i] = map(int, input().split())
    a = solve(H, W, Q, R, C)
    print(a)

if __name__ == '__main__':
    main()
