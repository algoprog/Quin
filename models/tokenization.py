import numpy as np


def get_segments(sentence):
    sentence_segments = []
    temp = []
    i = 0
    for token in sentence.split(" "):
        temp.append(i)
        if token == "[SEP]":
            i += 1
    sentence_segments.append(temp)
    return sentence_segments


def tokenize(text, max_length, tokenizer, second_text=None):
    if second_text is None:
        sentence = "[CLS] " + " ".join(tokenizer.tokenize(text.replace('[^\w\s]+|\n', ''))[:max_length-2]) + " [SEP]"
    else:
        text = tokenizer.tokenize(text.replace('[^\w\s]+|\n', ''))
        second_text = tokenizer.tokenize(second_text.replace('[^\w\s]+|\n', ''))
        while len(text) + len(second_text) > max_length - 3:
            if len(text) > len(second_text):
                text.pop()
            else:
                second_text.pop()
        sentence = "[CLS] " + " ".join(text) + " [SEP] " + " ".join(second_text) + " [SEP]"

    # generate masks
    # bert requires a mask for the words which are padded.
    # Say for example, maxlen is 100, sentence size is 90. then, [1]*90 + [0]*[100-90]
    sentence_mask = [1] * len(sentence.split(" ")) + [0] * (max_length - len(sentence.split(" ")))

    # generate input ids
    # if less than max length provided then the words are padded
    if len(sentence.split(" ")) != max_length:
        sentence_padded = sentence + " [PAD]" * (max_length - len(sentence.split(" ")))
    else:
        sentence_padded = sentence

    sentence_converted = tokenizer.convert_tokens_to_ids(sentence_padded.split(" "))

    # generate segments
    # for each separation [SEP], a new segment is converted
    sentence_segment = get_segments(sentence_padded)

    # convert list into tensor integer arrays and return it
    # return sentences_converted,sentences_segment, sentences_mask
    """
    return [tf.cast(sentence_converted, tf.int32),
            tf.cast(sentence_segment, tf.int32),
            tf.cast(sentence_mask, tf.int32)]
    """
    return [np.asarray(sentence_converted, dtype=np.int32).squeeze(),
            np.asarray(sentence_segment, dtype=np.int32).squeeze(),
            np.asarray(sentence_mask, dtype=np.int32).squeeze()]
