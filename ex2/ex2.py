'''
HW2 for Probabilistic Models Course

Students: Daniela Stepanov Daniel Juravski 308221720 ---

29/12

'''

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
        self.dev_events = {}
        self.test_events = {}
        self.dev4Lid_train = []
        self.dev4Lid_val = []
        self.dev4Lid_train_dict = {}
        self.dev4Lid_val_dict = {}
        self.dev4HO_T = []
        self.dev4HO_H = []
        self.dev4HO_T_dict = {}
        self.dev4HO_H_dict = {}
        self.dict_r_tr = {}
        self.dict_r_nr = {}
        self.dict_r_keys = {}
        self.dev4HO_T_set = set()
        self.dev4HO_H_set = set()
        self.best_lam = 0.0


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


def printOutput(output_file_name, table_items):
    with open(output_file_name, 'w') as f:
        f.write("#Students" + '\t'+ "Daniela Stepanov" + '\t' + "Daniel Juravski" + '\t' + '308221720' + '\t' + '---' + '\n')

        for key, val in OUTPUTS.items():
            out_print = "#" + key + '\t' + str(val) + '\n'
            f.write(out_print)
            if DEV:
                print(out_print, end="")

        for i, (fl, fho, nr, tr) in enumerate(table_items):
            f.write(str(i) + '\t' + str(fl) + '\t' + str(fho) + '\t' + str(nr) + '\t' + str(tr) + '\n')



def devSetPreProcessing():
    """
    impl. Development set preprocessing part of the pdf
    iterate the events dict, summerise the keys instances
    """
    data.S = sum([val for key, val in data.dev_events.items()])

    setOutputInfo("Output7", data.S)


def getWordLidstone(input_word, lam):
    p_lid = (getWordLidFreq(input_word) + lam) / (len(data.dev4Lid_train) + lam * V)

    return p_lid


def getWordLidFreq(input_word):
    input_word_freq = data.dev4Lid_train_dict[input_word] if input_word in data.dev4Lid_train_dict else 0

    return input_word_freq


def getWordHOFreq(input_word):
    input_word_freq = data.dev4HO_T_dict[input_word] if input_word in data.dev4HO_T_dict else 0

    return input_word_freq


def getPerplexity(smoothF, lam):
    sum = 0

    for word in data.dev4Lid_val:
        p = smoothF(word, lam)
        if p == 0:
            log_p = -float("inf")  # log(0) == -inf
        else:
            log_p = np.log2(p)
        sum += log_p

    avg = (-1/len(data.dev4Lid_val)) * sum
    prep = np.power(2, avg)

    return prep

def getPerplexityTest(smoothF, lam):
    sum = 0

    for word in data.test_tokens:
        p = smoothF(word, lam)
        if p == 0:
            log_p = -float("inf")  # log(0) == -inf
        else:
            log_p = np.log2(p)
        sum += log_p

    avg = (-1/len(data.test_tokens)) * sum
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
        cand_perp = getPerplexity(getWordLidstone, curr_lam)
        if cand_perp < min_perp:
            min_perp = cand_perp
            best_lam = curr_lam

        curr_lam += 0.01

    best_lam = round(best_lam, 2)
    data.best_lam = best_lam
    return best_lam, min_perp


def lidstonePart(input_word):
    """
    Split dev into train (90%) and val(10%) sets of the total words (not events).
    Meaning we count also the number of times each word appears until reaching 90%.
    """
    data.dev4Lid_train = data.dev_tokens[0: round(0.90*data.S)]
    data.dev4Lid_val = data.dev_tokens[round(0.90*data.S):]
    data.dev4Lid_train_set = set(data.dev4Lid_train)
    data.dev4Lid_val_set = set(data.dev4Lid_val)
    data.dev4Lid_train_dict = list2dict(data.dev4Lid_train)
    data.dev4Lid_val_dict = list2dict(data.dev4Lid_val)

    setOutputInfo("Output8", len(data.dev4Lid_val))
    setOutputInfo("Output9", len(data.dev4Lid_train))
    setOutputInfo("Output10", len(data.dev4Lid_train_set))
    setOutputInfo("Output11", getWordLidFreq(input_word))
    setOutputInfo("Output12", getWordLidFreq(input_word) / len(data.dev4Lid_train))
    setOutputInfo("Output13", getWordLidFreq(UNK_WORD) / len(data.dev4Lid_train))

    p_lid = getWordLidstone(input_word, lam=0.10)
    setOutputInfo("Output14", p_lid)

    p_lid = getWordLidstone(UNK_WORD, lam=0.10)
    setOutputInfo("Output15", p_lid)

    perplexity = getPerplexity(getWordLidstone, 0.01)
    setOutputInfo("Output16", perplexity)

    perplexity = getPerplexity(getWordLidstone, 0.10)
    setOutputInfo("Output17", perplexity)

    perplexity = getPerplexity(getWordLidstone, 1.00)
    setOutputInfo("Output18", perplexity)

    best_lam, min_perp = findBestLam()
    setOutputInfo("Output19", best_lam)
    setOutputInfo("Output20", min_perp)


def getTRandNR(r):
    r_words = {}

    # find words in T with of r
    for word, freq in data.dev4HO_T_dict.items():
        if freq == r:
            r_words[word] = 0

    # get the below words freq
    sum = 0
    for word, freq in r_words.items():
        r_words[word] = data.dev4HO_H_dict
        sum += r_words[word]

    return sum


def initHODicts():
    dict_r_tr = dict()  # {r -> tr}
    dict_r_nr = dict()  # {r -> Nr}
    r_set = set(data.dev4HO_T_dict.values())  # (r1, r2, ..., r_n)
    dict_r_keys = dict()  # {r_x -> [word1, word2, .., word_n]}


    for k,v in data.dev4HO_T_dict.items():
        if v in dict_r_keys:
            dict_r_keys[v].append(k)
        else:
            dict_r_keys[v] = [k]

    dict_r_tr[0] = 0
    for r in r_set:
        tr = 0
        for word in dict_r_keys[r]:
            if word in data.dev4HO_H_dict:
                tr += data.dev4HO_H_dict[word]
        dict_r_tr[r] = tr

        nr = len(dict_r_keys[r])
        dict_r_nr[r] = nr

    for word in data.dev4HO_H_set:
        if word not in data.dev4HO_T_dict:
            dict_r_tr[0] += data.dev4HO_H_dict[word]

    dict_r_nr[0] = V - len(data.dev4HO_T_dict)
    data.dict_r_tr = dict_r_tr
    data.dict_r_nr = dict_r_nr
    data.dict_r_keys = dict_r_keys


def getWordHO(input_word, lam=False):
    word_r = getWordHOFreq(input_word)
    tr = data.dict_r_tr[word_r]
    nr = data.dict_r_nr[word_r]

    p = (tr / nr) / len(data.dev4HO_H)

    return p


def heldOutPart(input_word):
    data.dev4HO_T = data.dev_tokens[0: round(0.5*data.S)] #train data
    data.dev4HO_H = data.dev_tokens[round(0.5*data.S):] #heldout data
    data.dev4HO_T_set = set(data.dev4HO_T)
    data.dev4HO_H_set = set(data.dev4HO_H)
    data.dev4HO_T_dict = list2dict(data.dev4HO_T)
    data.dev4HO_H_dict = list2dict(data.dev4HO_H)

    setOutputInfo("Output21", len(data.dev4HO_T))
    setOutputInfo("Output22", len(data.dev4HO_H))

    initHODicts()

    p_ho = getWordHO(input_word)
    setOutputInfo("Output23", p_ho)

    p_ho = getWordHO(UNK_WORD)
    setOutputInfo("Output24", p_ho)

def testPart(input_word):
    data.test_dict = list2dict(data.test_tokens)

    setOutputInfo("Output25", len(data.test_tokens))

    perpL = getPerplexityTest(getWordLidstone, data.best_lam)
    setOutputInfo("Output26", perpL)

    perpHO = getPerplexityTest(getWordHO, data.best_lam)
    setOutputInfo("Output27", perpHO)

    bestP = 'H' if perpHO < perpL else 'L'
    setOutputInfo("Output28", bestP)

def tablePart():
    print(data.best_lam)
    fl = [round((i + data.best_lam) * len(data.dev4Lid_train) / (len(data.dev4Lid_train) + data.best_lam * V), 5) for i in range(10)]
    fho = [round(data.dict_r_tr[i] * len(data.dev4HO_T) / (len(data.dev4HO_H) * data.dict_r_nr[i]), 5) for i in range(10)]
    Nr = [data.dict_r_nr[i] for i in range(10)]
    Tr = [int(data.dict_r_tr[i]) for i in range(10)]

    setOutputInfo("Output29", "")
    return zip(fl, fho, Nr, Tr)


if __name__ == '__main__':

    dev_set_file_name, test_set_file_name, input_word, output_file_name = getArgs()  # 1-4
    init()  # 5-6
    loadSets(dev_set_file_name, test_set_file_name)
    devSetPreProcessing()  # 7
    lidstonePart(input_word)  # 8-13
    heldOutPart(input_word)  # 21-24

    # -Debug- p(x*)n0 + sum(all x > 0)p(x) = 1

    #lidstone
    Nr0 = (V - len(data.dev4Lid_train_dict))* getWordLidstone(UNK_WORD, lam=data.best_lam)
    debug_lidstone = 0
    for word in data.dev4Lid_train_dict: #sum of P for all x in train with p > 0.
        debug_lidstone += getWordLidstone(word, lam=data.best_lam)

    debug_lidstone += Nr0
    if round(debug_lidstone, 4) != 1:
        print("debug lidstone does not sum to 1 ", debug_lidstone)

    #print(round(debug_lidstone, 4))

    #Heldout
    Nr0 = float((V - len(data.dev4Lid_train_dict)) * getWordHO(UNK_WORD))
    debug_heldout = 0
    for word in data.dev4Lid_train_dict:  # sum of P for all x in train with p > 0.
        debug_heldout += float(getWordHO(word))
    debug_heldout += Nr0
    if round(debug_heldout, 5) != 1:
        print("debug heldout does not sum to 1 ", debug_heldout)

    testPart(input_word)
    table_items = tablePart()

    printOutput(output_file_name, table_items)



