#!/usr/bin/env python

import numpy as np
import sys

nums = sys.argv

def sum_digits(X):
    a = 0
    tot = 0
    while(a < len(X)):
        tot += int(X[a])
        a += 1
    print tot
        
b = 1
iter = 0
while(b < len(nums)):
    if len(nums) > b+1:
        if int(nums[b])**2 <= int(nums[b+1]):
            count = int(nums[b])
            while(count <= int(nums[b])**2):
                if count % int(nums[b]) == 0:
                    term = [int(i) for i in str(count)]
                    sum_digits(term)
                else:
                    pass
                count +=1
                iter += 1
            
        else:
            count = int(nums[b])
            while(count <= int(nums[b+1])):
                if count % int(nums[b]) == 0:
                    term = [int(i) for i in str(count)]
                    sum_digits(term)
                else:
                    pass
                count += 1
                iter += 1

    else:
        count = int(nums[b])
        while(count <= int(nums[b])**2):
            if count % int(nums[b]) == 0:
                term = [int(i) for i in str(count)]
                sum_digits(term)
            else:
                pass
            count +=1
            iter += 1
            if len(nums) == b+1 and len(nums) > 2:
                break
    b += 1
