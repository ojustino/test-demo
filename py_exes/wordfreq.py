import numpy as np

def wordfreq(filename):
    f1 = open(filename)
    xx1 = f1.readlines()

    speech = {}

    # create dictionary of relevant words (longer than 3 characters)
    for row in xx1: 
        row = row.split()
        for word in row:
            if word.isalpha(): #if word is totally alphabetic
                word = word.lower()
                if word in speech and len(word) > 3:
                    speech[word] += 1 # increment it
                if word not in speech and len(word) > 3:
                    speech[word] = 1  # add it to the dict
                else:
                    pass
                continue
            if word[0:-1].isalpha(): # if word is alpha but has punct at end
                word = word[0:-1].lower()
                if word in speech and len(word) > 4:
                    speech[word] += 1 # increment it
                if word not in speech and len(word) > 4:
                    speech[word] = 1  # add it to the dict
                else:
                    pass
                continue
            else:
                pass
    
    # now print sorted style
    #print sorted(speech, key=lambda x: x[1])
            
    print speech

wordfreq('ihaveadream.txt')


        
# make all words lowercase, add together words that were previously in different cases
# discard all punctuation and numbers
# x=
