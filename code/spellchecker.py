import re
from collections import Counter
import helpers as hlp

def words(text): return re.findall(r'\w+', text.lower())

def P(word, rank):
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - rank.get(word, 0)

def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    reductions = [re.sub(r'([{}])\1+'.format(i), r'\1', word) for i in set(word)]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


# That's it. If you have downloaded word2vec then you can start using this code.  Here are few examples of what it does.
# 
# correction('quikly') returns quickly
# 
# correction('israil') returns israel
# 
# correction('neighbour') returns neighbor

# If you like this notebook then please upvote (button at the top right).
