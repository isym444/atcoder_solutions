#!/usr/bin/env python3
# from typing import *



# def solve(H: str, W: str, N: str, a: List[str], b: List[str]) -> Any:
def solve(H, W, N, a, b):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    H, W, N = input().split()
    a = [None for _ in range(N)]
    b = [None for _ in range(N)]
    for i in range(N):
        a[i], b[i] = input().split()
    ans = solve(H, W, N, a, b)
    print(ans)  # TODO: edit here

if __name__ == '__main__':
    main()