#!/usr/bin/env python3
# usage: $ oj generate-input 'python3 generate.py'
# usage: $ oj generate-input --hack-actual=./a.out --hack-expected=./naive 'python3 generate.py' number_of_test_cases_you_want

import random
import string

def generate_random_string(min_length=1, max_length=50):
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_uppercase, k=length))

# Example usage:
random_string = generate_random_string()
print(random_string)
