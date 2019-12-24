import sys
import numpy as np
import math
import datetime
from collections import defaultdict


DEV = True
### Globals start ###
OUTPUTS = {}  # global dict of outputs
V = 300000  # given vocab size
UNK_WORD = "unseen-word"
### Globals end ###


class Data:
    def __init__(self):
        self.S = 0
        self.dev_tokens = 0
        self.test_tokens = 0
        self.train_set = set()
        self.val_set = set()
        self.dev4train = []
        self.dev4val = []
        self.dev_events = {}
        self.test_events = {}
        self.dev4train_dict = {}
        self.dev4val_dict = {}


data = Data()


def setOutputInfo(keyName, value):
    """
    Add key output value to the global OUTPUT dict
    """
    OUTPUTS[keyName] = value


def getArgs():
    """
    Get the 4 needed arguments for this ex.
    """

    if len(sys.argv) != 5:
        print("[ERROR]: The required input is: "
              "python ex2.py < development set filename > < test set filename > < INPUT WORD > < output filename >")
        exit(1)

    dev_set_file_name = sys.argv[1]
    test_set_file_name = sys.argv[2]
    input_word = sys.argv[3]
    output_file_name = sys.argv[4]

    setOutputInfo("Output1", dev_set_file_name)
    setOutputInfo("Output2", test_set_file_name)
    setOutputInfo("Output3", input_word)
    setOutputInfo("Output4", output_file_name)

    return dev_set_file_name, test_set_file_name, input_word, output_file_name


def init():
    """
    impl. init part (specifically (e) and (f)) of the pdf
    """
    setOutputInfo("Output5", V)  # (e), 300,000

    p_uniform = 1/V
    setOutputInfo("Output6", p_uniform)  # (f), uniform distribution


def events2Dict(set_file_name, tokens):
    """
    scan the articles txt, every word that appears in the article, push into the dict and increase it's counter by 1.
    """

    d = defaultdict(int)
    with open(set_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # look for article line
            if (line[0] == '<' and line[-2:] == '>\n') or (line == '\n'):
                pass
            else:
                # the line is article
                for word in line.split():
                    tokens.append(word)
                    if word in d:
                        d[word] += 1
                    else:
                        d[word] = 1

    return d, tokens


def loadSets(dev_set_file_name, test_set_file_name):
    """
    Load the data sets files into event objects (dicts)
    """
    data.dev_events, data.dev_tokens = events2Dict(dev_set_file_name, [])
    data.test_events, data.test_tokens = events2Dict(test_set_file_name, [])


def printOutput(output_file_name):
    with open(output_file_name, 'w') as f:
        for key, val in OUTPUTS.items():
            out_print = "#" + key + '\t' + str(val) + '\n'
            f.write(out_print)
            if DEV:
                print(out_print, end="")


def devSetPreProcessing():
    """
    impl. Development set preprocessing part of the pdf
    iterate the events dict, summerise the keys instances
    """
    data.S = sum([val for key, val in data.dev_events.items()])

    setOutputInfo("Output7", data.S)


def getWordLidstone(input_word, lam):
    p_lid = (getWordFreq(input_word) + lam) / (len(data.dev4train) + lam * V)

    return p_lid


def getWordFreq(input_word):
    input_word_freq = data.dev4train_dict[input_word] if input_word in data.dev4train_dict else 0

    return input_word_freq


def getPerplexity(lam):
    sum = 0

    for word in data.dev4val:
        p = getWordLidstone(word, lam)
        if p == 0:
            log_p = -float("inf")  # log(0) == -inf
        else:
            log_p = np.log2(p)
        sum += log_p

    avg = (-1/len(data.dev4val)) * sum
    prep = np.power(2, avg)

    return prep


def list2dict(list):
    d = defaultdict(int)

    for word in list:
        if word in d:
            d[word] += 1
        else:
            d[word] = 1

    return d


def findBestLam():
    min_perp = float("inf")
    best_lam = 0
    curr_lam = 0

    while (curr_lam <= 2):
        cand_perp = getPerplexity(curr_lam)
        if cand_perp < min_perp:
            min_perp = cand_perp
            best_lam = curr_lam

        curr_lam += 0.01

    best_lam = round(best_lam, 2)

    return best_lam, min_perp


def lidstonePart(input_word):
    """
    Split dev into train (90%) and val(10%) sets of the total words (not events).
    Meaning we count also the number of times each word appears until reaching 90%.
    """
    data.dev4train = data.dev_tokens[0: round(0.90*data.S)]
    data.dev4val = data.dev_tokens[round(0.90*data.S):]
    data.dev4train_set = set(data.dev4train)
    data.dev4val_set = set(data.dev4val)
    data.dev4train_dict = list2dict(data.dev4train)
    data.dev4val_dict = list2dict(data.dev4val)

    setOutputInfo("Output8", len(data.dev4val))
    setOutputInfo("Output9", len(data.dev4train))
    setOutputInfo("Output10", len(data.dev4train_set))
    setOutputInfo("Output11", getWordFreq(input_word))
    setOutputInfo("Output12", getWordFreq(input_word) / len(data.dev4train))
    setOutputInfo("Output13", getWordFreq(UNK_WORD) / len(data.dev4train))

    p_lid = getWordLidstone(input_word, lam=0.10)
    setOutputInfo("Output14", p_lid)

    p_lid = getWordLidstone(UNK_WORD, lam=0.10)
    setOutputInfo("Output15", p_lid)

    perplexity = getPerplexity(0.01)
    setOutputInfo("Output16", perplexity)

    perplexity = getPerplexity(0.10)
    setOutputInfo("Output17", perplexity)

    perplexity = getPerplexity(1.00)
    setOutputInfo("Output18", perplexity)

    best_lam, min_perp = findBestLam()
    setOutputInfo("Output19", best_lam)
    setOutputInfo("Output20", min_perp)


if __name__ == '__main__':
    if DEV:
        print("Start: ")
        print(datetime.datetime.now().time())

    dev_set_file_name, test_set_file_name, input_word, output_file_name = getArgs()  # 1-4
    init()  # 5-6
    loadSets(dev_set_file_name, test_set_file_name)
    devSetPreProcessing()  # 7
    lidstonePart(input_word)  # 8-13

    printOutput(output_file_name)

    if DEV:
        print("\nEnd: ")
        print(datetime.datetime.now().time())


