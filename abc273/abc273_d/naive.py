#!/usr/bin/env python3
# from typing import *



# def solve(n: int, a: List[int]) -> Tuple[List[str], List[str]]:
def solve(n, a):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    # failed to analyze input format
    n = int(input())  # TODO: edit here
    a = list(map(int, input().split()))  # TODO: edit here
    R, C = solve(n, a)
    for i in range(Q):
        print(R[i], C[i])

if __name__ == '__main__':
    main()
