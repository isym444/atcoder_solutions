#!/usr/bin/env python3
# from typing import *



# def solve(A: int, op: str, B: int) -> int:
def solve(A, op, B):
    pass  # TODO: edit here

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    import sys
    tokens = iter(sys.stdin.read().split())
    A = int(next(tokens))
    op = next(tokens)
    B = int(next(tokens))
    assert next(tokens, None) is None
    a = solve(A, op, B)
    print(a)

if __name__ == '__main__':
    main()