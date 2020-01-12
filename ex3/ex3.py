import sys
import os
import numpy as np
import copy
from time import gmtime, strftime
import matplotlib.pyplot as plt


EPS = 0.001
LAMBDA = 0.1
K = 10
ITERATIONS = 30


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
        self.V_dict = dict()  # dict of words and freq
        self.w2i = dict()  # mapping of V_dict words to indexes
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
        V_dict_doc = dict()
        for word in doc.text:
            if word not in V_dict:
                doc.text.remove(word)
            else:
                if word in V_dict_doc:
                    V_dict_doc[word] += 1
                else:
                    V_dict_doc[word] = 1
        doc.n_t = len(doc.text)
        doc.n_tk = V_dict_doc

    return documents, V_dict


def initData(documents, V_dict):
    print("[INFO] Initializing matrices and vectors")
    data = Data()
    N = len(documents)
    V = len(V_dict)
    data.N = N
    data.V = V
    data.V_dict = V_dict

    # init w2i mapping
    for i, w in enumerate(V_dict):
        data.w2i[w] = i

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


def iterateAlpha(data):
    time = strftime("%H:%M:%S", gmtime())
    print("[INFO] [iterateAlpha] start time: {0}".format(time))
    for i in range(data.C):
        # for doc in documents:
        sum = np.sum(data.w[:,i])
        sum = sum / data.N
        if sum < EPS:
            data.alpha[i] = EPS
        data.alpha[i] = sum

    data.alpha /= np.sum(data.alpha)

    time = strftime("%H:%M:%S", gmtime())
    print("[INFO] [iterateAlpha] end time: {0}".format(time))
    print("[INFO] [iterateAlpha] sum of Alpha vector: {0} (should be {1})".format(np.sum(data.alpha), 1))


def iterateP(data, documents):
    time = strftime("%H:%M:%S", gmtime())
    print("[INFO] [iterateP] start time: {0}".format(time))
    for i in range(data.C):
        p_i = np.random.random(data.V)  # random vector
        sum_p_i = 0
        for n in range(data.N):
            p_i = np.full(data.V, LAMBDA)  # random vector
            w_ti = data.w[n, i]
            for word, freq in documents[n].n_tk.items():
                n_tk = freq
                p_i[data.w2i[word]] = w_ti * n_tk + LAMBDA
            sum_p_i = np.sum(p_i)

        p_ik = p_i/sum_p_i
        data.P[i,:] = p_ik

    time = strftime("%H:%M:%S", gmtime())
    print("[INFO] [iterateP] end time: {0}".format(time))
    print("[INFO] [iterateP] sum of P matrix: {0} (should be {1})".format(np.sum(data.P), data.C))


def mStep(data, documents):
    iterateAlpha(data)
    iterateP(data, documents)


def getDocumentWordFreq(document, k_word):
    if k_word not in document.n_tk:
        freq = 0
    else:
        freq = document.n_tk[k_word]

    return freq


def calcZ(data, documents, n, i):
    ln_alpha_i = np.log(data.alpha[i])
    sum = 0
    for word, freq in documents[n].n_tk.items():
        n_tk = freq
        p_ik = data.P[i, data.w2i[word]]
        value = n_tk * np.log(p_ik)
        sum += value

    value = ln_alpha_i + sum

    return value


def iterateW(data, documents):
    time = strftime("%H:%M:%S", gmtime())
    print("[INFO] [iterateW] start time: {0}".format(time))
    for n in range(data.N):
        z_j = np.random.random(data.C)  # random vector
        for i in range(data.C):
            z_i = calcZ(data, documents, n, i)
            z_j[i] = z_i
        m = np.max(z_j)
        sum_j = 0
        e_zi = np.random.random(data.C)  # random vector
        for i in range(data.C):
            if z_j[i] - m < -K:
                value = 0
            else:
                value = np.exp(z_j[i]-m)
            e_zi[i] = value
            sum_j += value
        cand_w_ti = e_zi/sum_j
        data.w[n,:] = cand_w_ti

    time = strftime("%H:%M:%S", gmtime())
    print("[INFO] [iterateW] end time: {0}".format(time))
    print("[INFO] [iterateW] sum of w matrix: {0} (should be {1})".format(np.sum(data.w), data.N))


def eStep(data, documents):
    iterateW(data, documents)


def calcLL(data, documents):
    time = strftime("%H:%M:%S", gmtime())
    print("[INFO] [calcLL] start time: {0}".format(time))
    sum_t = 0
    for n in range(data.N):
        z_j = np.random.random(data.C)  # random vector
        for i in range(data.C):
            z_i = calcZ(data, documents, n, i)
            z_j[i] = z_i
        m = np.max(z_j)
        sum_j = 0
        for i in range(data.C):
            if z_j[i] - m >= -K:
                value = np.exp(z_j[i]-m)
                sum_j += value
        value = m + np.log(sum_j)
        sum_t += value

    time = strftime("%H:%M:%S", gmtime())
    print("[INFO] [calcLL] start time: {0}".format(time))

    return sum_t


def makeLLDashboard(ll_prog):
    ll = np.asarray(ll_prog)
    plt.plot(ll)
    plt.ylabel('LL')
    plt.xlabel('Iteration')
    plt.title('Log-Likelihood relative to Iteration number')
    plt.tight_layout()

    plt.show()
    plt.savefig('LL')


def makePerplexityDashboard(ll_prog, data):
    p = np.exp(-np.asarray(ll_prog)/data.N)
    plt.plot(p)
    plt.ylabel('Perplexity')
    plt.xlabel('Iteration')
    plt.title('Perplexity relative to Iteration number')
    plt.tight_layout()

    plt.show()
    plt.savefig('LL')


def makeStats(data, documents, ll_prog):
    makeLLDashboard(ll_prog)
    # makePerplexityDashboard(ll_prog, data)


if __name__ == '__main__':
    develop_file_name = getArgs()
    documents, V_dict = loadDataSet(develop_file_name)
    data = initData(documents, V_dict)
    mStep(data, documents)

    print("[INFO] Starting EM")
    ll_prog = []
    for iter in range(ITERATIONS):
        start_time = strftime("%H:%M:%S", gmtime())
        eStep(data, documents)
        mStep(data, documents)
        ll = calcLL(data, documents)
        ll_prog.append(ll)

        end_time = strftime("%H:%M:%S", gmtime())
        print("\tIter:{0} s_time: {1}".format(iter, start_time))
        print("\tIter:{0} e_time: {1} ll:{2}".format(iter, end_time, ll))

    print("[INFO] Let's make some statistics")
    makeStats(data, documents, ll_prog)




    pass

