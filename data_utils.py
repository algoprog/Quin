import json
import logging
import tqdm

from typing import Union, List


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(self, guid: str, texts: List[str], label: Union[int, float]):
        """
        Creates one InputExample with the given texts, guid and label
        str.strip() is called on both texts.
        :param guid
            id for the example
        :param texts
            the texts for the example
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = [text.strip() for text in texts]
        self.label = label

    def get_texts(self):
        return self.texts

    def get_label(self):
        return self.label


class LoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def get_examples(filename, max_examples=0):
    examples = []
    id = 0
    with open(filename, encoding='utf8') as file:
        for j, line in enumerate(file):
            line = line.rstrip('\n')
            sample = json.loads(line)
            label = sample['label']
            guid = "%s-%d" % (filename, id)
            id += 1
            if label == 'entailment':
                label = 0
            elif label == 'contradiction':
                label = 1
            else:
                label = 2
            examples.append(InputExample(guid=guid,
                                         texts=[sample['s1'], sample['s2']],
                                         label=label))
            if 0 < max_examples <= len(examples):
                break
    return examples


def get_qa_examples(filename, max_examples=0, dev=False):
    examples = []
    id = 0
    with open(filename, encoding='utf8') as file:
        for j, line in enumerate(file):
            line = line.rstrip('\n')
            sample = json.loads(line)
            label = sample['relevant']
            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid,
                                         texts=[sample['question'], sample['answer']],
                                         label=label))
            if not dev:
                if label == 1:
                    for _ in range(13):
                        examples.append(InputExample(guid=guid,
                                                     texts=[sample['question'], sample['answer']],
                                                     label=label))
            if 0 < max_examples <= len(examples):
                break
    return examples


def map_label(label):
    labels = {"relevant": 0, "irrelevant": 1}
    return labels[label.strip().lower()]


def get_pair_input(tokenizer, sent1, sent2, max_len=256):
    text = "[CLS] {} [SEP] {} [SEP]".format(sent1, sent2)

    tokenized_text = tokenizer.tokenize(text)[:max_len]
    indexed_tokens = tokenizer.encode(text)[:max_len]

    segments_ids = []
    sep_flag = False
    for i in range(len(tokenized_text)):
        if tokenized_text[i] == '[SEP]' and not sep_flag:
            segments_ids.append(0)
            sep_flag = True
        elif sep_flag:
            segments_ids.append(1)
        else:
            segments_ids.append(0)
    return indexed_tokens, segments_ids


def build_batch(tokenizer, text_list, max_len=256):
    token_id_list = []
    segment_list = []
    attention_masks = []
    longest = -1

    for pair in text_list:
        sent1, sent2 = pair
        ids, segs = get_pair_input(tokenizer, sent1, sent2, max_len=max_len)
        if ids is None or segs is None:
            continue
        token_id_list.append(ids)
        segment_list.append(segs)
        attention_masks.append([1] * len(ids))
        if len(ids) > longest:
            longest = len(ids)

    if len(token_id_list) == 0:
        return None, None, None

    # padding
    assert (len(token_id_list) == len(segment_list))
    for ii in range(len(token_id_list)):
        token_id_list[ii] += [0] * (longest - len(token_id_list[ii]))
        attention_masks[ii] += [1] * (longest - len(attention_masks[ii]))
        segment_list[ii] += [1] * (longest - len(segment_list[ii]))

    return token_id_list, segment_list, attention_masks
