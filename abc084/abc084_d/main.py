#!/usr/bin/env python3
# from typing import *



# def solve(Q: int, l: List[int], r: List[int]) -> List[str]:
def solve(Q, l, r):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    Q = int(input())
    l = [None for _ in range(Q)]
    r = [None for _ in range(Q)]
    for i in range(Q):
        l[i], r[i] = map(int, input().split())
    a = solve(Q, l, r)
    for i in range(Q):
        print(a[i])

if __name__ == '__main__':
    main()
