#!/usr/bin/env python

import numpy as np
import sys

nums = sys.argv

def sum_digits(X):
    a = 1
    tot = 0
    while(a < len(X)):
        tot += int(X[a])
        a += 1
    print tot
        
c = 2
b = c-1
iter = 0
while(c <= len(nums)-1):
    if int(nums[b])**2 < int(nums[c]):
          count = nums[b]
          while(int(count) < int(nums[b])**2):
              if int(count) % int(nums[b]) == 0:
                  term = [int(i) for i in count]
                  sum_digits(term)
              else:
                  pass
              iter = int(count)
              iter += 1

    if int(nums[c]) < int(nums[b])**2:
        count = nums[b]
        while(int(count) < int(nums[c])):
            if int(count) % int(nums[b]) == 0:
                term = [int(i) for i in count]
                sum_digits(term)
            else:
                pass
            iter = int(count)
            iter += 1
    b += 1
    c += 1
