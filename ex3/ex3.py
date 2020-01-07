import sys
import os
import numpy as np
import copy


class Document:
    def __init__(self):
        self.text = []
        self.gold_topics = []
        self.n_t = 0  # length of doc
        self.n_tk = dict()  # freq of words in doc


class Data:
    def __init__(self):
        self.C = 9  # number of clusters (topics)
        self.N = 0  # number of documents
        self.V = 0  # vocab size
        self.alpha = None  # vector of P(C_i)
        self.P = None  # matrix of P(W_k|C_i)
        self.w = None  # matrix of P(C_i|Y_t)


def getArgs():
    """
    Get optional arg. If no arg was supplied, search for data set file at the current location
    """

    if len(sys.argv) == 2:
        print("[INFO] Loading argument")
        develop_file_name = sys.argv[1]
    else:
        develop_file_name = 'develop.txt'
        if not (os.path.exists(develop_file_name)):
            print("[ERROR] No arguments were supplied and {0} file wasn't found in the current directory, exiting.".format(develop_file_name))
            exit(1)
        print("[INFO] No arguments were supplied, using {0} in the current directory".format(develop_file_name))

    return develop_file_name


def makeCleanV(V_dict):
    V = dict()
    for word, freq in V_dict.items():
        V[word] = 0

    return V


def loadDataSet(develop_file_name):
    """
    1. load documents from file
    2. filter rare words & update meta
    :param develop_file_name:
    :return:
    """
    print("[INFO] Loading documents from data set")
    documents = []  # list of Document objects
    cand_V = dict()  # not a official V, out of this V, we will analyse the rare words

    with open(develop_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '<' and line[-2:] == '>\n':
                # that line is a topic line
                line_details = line.split()
                topics = line_details[2:]
            elif line == '\n':
                # that line is just a blank line
                pass
            else:
                # the line is an article
                splited_line = line.split()
                for word in splited_line:
                    if word in cand_V:
                        cand_V[word] += 1
                    else:
                        cand_V[word] = 1

                # we can declare a new document now
                document = Document()
                document.text = splited_line
                document.gold_topics = topics
                documents.append(document)

    # filter rare words in V_dict
    print("[INFO] Filtering rare words")
    V_dict = copy.deepcopy(cand_V)
    for word, freq in cand_V.items():
        if freq <= 3:
            del V_dict[word]

    # update documents meta
    print("[INFO] Finalizing documents initialization")
    for doc in documents:
        V_dict_doc = makeCleanV(V_dict)  # updated V with frequencies of 0
        for word in doc.text:
            if word not in V_dict:
                doc.text.remove(word)
            else:
                V_dict_doc[word] += 1
        doc.n_t = len(doc.text)
        doc.n_tk = V_dict_doc

    return documents


def initData(documents):
    print("[INFO] Initializing matrices and vectors")
    data = Data()
    N = len(documents)
    V = len(documents[0].n_tk)
    data.N = N
    data.V = V

    # init *alpha* vector
    alpha = np.random.random(data.C)  # random vector
    alpha /= alpha.sum()  # normalize to sum-to-1 vector
    data.alpha = alpha

    # init *P* matrix
    P = []
    for i in range(data.C):
        vec = np.random.random(data.V)  # random vector
        vec /= vec.sum()  # normalize to sum-to-1 vector
        P.append(vec)
    P = np.asarray(P)
    data.P = P

    # init *w* matrix
    # by the ex. definition, the initial div need to be by mod 9
    w = []
    for i in range(data.N):
        vec = np.random.random(data.C)  # random vector
        vec[i % data.C] += 1  # add 1 to insure the highest prob
        vec /= vec.sum()  # normalize to sum-to-1 vector
        w.append(vec)
    w = np.asarray(w)
    data.w = w

    return data


if __name__ == '__main__':
    develop_file_name = getArgs()
    documents = loadDataSet(develop_file_name)
    data = initData(documents)

    pass

