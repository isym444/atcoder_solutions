#!/usr/bin/env python3
# usage: $ oj generate-input 'python3 generate.py'
# usage: $ oj generate-input --hack-actual=./a.out --hack-expected=./naive 'python3 generate.py' number_of_test_cases_you_want
import random

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    R = random.randint(1, 10)  # TODO: edit here
    X = random.randint(0, 10)  # TODO: edit here
    Y = random.randint(0, 10)  # TODO: edit here
    print(R, end=" ")
    print(X, end=" ")
    print(Y, end=" ")

if __name__ == "__main__":
    main()
