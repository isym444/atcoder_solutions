#!/usr/bin/env python3
# from typing import *


# def solve(n: int, a: List[int]) -> Tuple[int, List[int], List[int], int]:
def solve(n, a):
    pass  # TODO: edit here


# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    # failed to analyze input format
    n = int(input())  # TODO: edit here
    a = list(map(int, input().split()))  # TODO: edit here
    a, b, c, d = solve(n, a)
    print(a)
    for i in range(a):
        print(b[i])
        print(c[i])
    print(d)


if __name__ == '__main__':
    main()