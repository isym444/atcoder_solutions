#!/usr/bin/env python3
# usage: $ oj generate-input 'python3 generate.py'
# usage: $ oj generate-input --hack-actual=./a.out --hack-expected=./naive 'python3 generate.py' number_of_test_cases_you_want
import random

# generated by oj-template v4.8.1 (https://github.com/online-judge-tools/template-generator)
def main():
    H = random.randint(1, 15)  # TODO: edit here
    W = random.randint(1, 15)  # TODO: edit here
    print(H, end=" ")
    print(W)
    for i in range(H):
        for j in range(W):
            selector = random.randint(1,2)
            if(selector==1):
                print("#", end="")
            else:
                print(".", end="")
        print()

if __name__ == "__main__":
    main()
