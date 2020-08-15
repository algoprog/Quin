import math
import re
import nltk


def is_question(query):
    if re.match(r'^(who|when|what|why|is|are|was|were|do|does|did|how)', query) or query.endswith('?'):
        return True

    pos_tags = nltk.pos_tag(nltk.word_tokenize(query))
    for tag in pos_tags:
        if tag[1].startswith('VB'):
            return False

    return True


def softmax(vector):
    sum = 0
    for v in vector:
        sum += math.exp(v)
    new_vector = []
    for v in vector:
        new_vector.append(math.exp(v) / sum)
    return new_vector