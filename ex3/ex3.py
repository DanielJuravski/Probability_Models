'''
Students: Daniela Stepanov Daniel Juravski 308221720 206082323

HW3 for Probabilistic Models Course


14/01

'''

import sys
import os
import numpy as np
import copy
from time import gmtime, strftime
from datetime import datetime, date
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import seaborn as sns
np.random.seed(0)

EPS = 0.001
LAMBDA = 1.0
K = 10
ITERATIONS = 20


class Document:
    def __init__(self):
        self.text = []
        self.gold_topics = []
        self.n_t = 0  # length of doc
        self.n_tk = dict()  # freq of words in doc
        self.pred_cluster = -1


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
        self.topic2index = {}
        self.index2topic = {}
        self.cluster_freq = {}


def getArgs():
    """
    Get optional arg. If no arg was supplied, search for data set file at the current location
    """

    if len(sys.argv) == 3:
        print("[INFO] Loading argument")
        develop_file_name = sys.argv[1]
        topics_file_name = sys.argv[2]
    else:
        develop_file_name = 'develop.txt'
        topics_file_name = 'topics.txt'
        if not (os.path.exists(develop_file_name) and os.path.exists(topics_file_name)):
            print("[ERROR] No arguments were supplied, file wasn't found in the current directory, exiting.".format())
            exit(1)
        print("[INFO] No arguments were supplied, using files in the current directory".format())

    return develop_file_name, topics_file_name


def setTopicsAndClusters(data, documents, topics_file_name):
    cluster_freq = {}  # cluster index to freq
    index2topic = {}
    topic2index = {}
    with open(topics_file_name, 'r') as f:
        lines = f.readlines()
        lines = [line.strip('\n') for line in lines]
        counter = 0
        for line in lines:
            if not line: continue
            # topics_freq[line] = 0
            index2topic[counter] = line
            topic2index[line] = counter
            cluster_freq[counter] = 0
            counter += 1

    # set cluster predictions to doc
    for n in range(data.N):
        pred_cluster = np.argmax(data.w[n])
        documents[n].pred_cluster = pred_cluster
        cluster_freq[pred_cluster] += 1

    data.topic2index = topic2index
    data.index2topic = index2topic
    data.cluster_freq = cluster_freq

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
                topics[-1] = topics[-1][:-1]
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
    s_time = strftime("%H:%M:%S", gmtime())
    for i in range(data.C):
        # for doc in documents:
        sum = np.sum(data.w[:,i])
        sum = sum / data.N
        if sum < EPS:
            data.alpha[i] = EPS
        data.alpha[i] = sum

    data.alpha /= np.sum(data.alpha)

    e_time = strftime("%H:%M:%S", gmtime())
    print("[INFO] [iterateAlpha] time: {0}".format(datetime.strptime(e_time, '%H:%M:%S') - datetime.strptime(s_time, '%H:%M:%S')))
    print("[INFO] [iterateAlpha] sum of Alpha vector: {0} (should be {1})".format(np.sum(data.alpha), 1))


def iterateP(data, documents):
    s_time = strftime("%H:%M:%S", gmtime())
    for i in range(data.C):
        p_ik = np.full(data.V, 0)
        for n in range(data.N):
            w_ti = data.w[n, i]
            for word, freq in documents[n].n_tk.items():
                n_tk = freq
                p_ik[data.w2i[word]] += w_ti * n_tk

        p_ik = p_ik + LAMBDA
        p_ik = p_ik / np.sum(p_ik)
        data.P[i,:] = p_ik

    e_time = strftime("%H:%M:%S", gmtime())
    print("[INFO] [iterateP] time: {0}".format(datetime.strptime(e_time, '%H:%M:%S') - datetime.strptime(s_time, '%H:%M:%S')))
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
    s_time = strftime("%H:%M:%S", gmtime())
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

    e_time = strftime("%H:%M:%S", gmtime())
    print("[INFO] [iterateW] time: {0}".format(datetime.strptime(e_time, '%H:%M:%S') - datetime.strptime(s_time, '%H:%M:%S')))
    print("[INFO] [iterateW] sum of w matrix: {0} (should be {1})".format(np.sum(data.w), data.N))


def eStep(data, documents):
    iterateW(data, documents)


def calcLL(data, documents):
    s_time = strftime("%H:%M:%S", gmtime())
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

    e_time = strftime("%H:%M:%S", gmtime())
    print("[INFO] [calcLL] time: {0}".format(datetime.strptime(e_time, '%H:%M:%S') - datetime.strptime(s_time, '%H:%M:%S')))

    return sum_t


def makeLLDashboard(ll_prog):
    ll = np.asarray(ll_prog)
    plt.plot(ll)
    plt.ylabel('LL')
    plt.xlabel('Iteration')
    plt.title('Log-Likelihood relative to Iteration number')
    plt.tight_layout()

    plt.savefig('LL')
    plt.clf()
    # plt.show()


def makePerplexityDashboard(ll_prog, data, documents):
    sum_ntk = 0
    for n in range(data.N):
        n_tk = documents[n].n_t
        sum_ntk += n_tk
    p = np.exp(-np.asarray(ll_prog)/(sum_ntk))
    plt.plot(p)
    plt.ylabel('Perplexity')
    plt.xlabel('Iteration')
    plt.title('Perplexity relative to Iteration number')
    plt.tight_layout()

    plt.savefig('Perplexity.png')
    plt.clf()
    # plt.show()


def makeStats(data, documents, ll_prog):
    makeLLDashboard(ll_prog)
    print("LL progress:{0}".format(ll_prog))

    makePerplexityDashboard(ll_prog, data, documents)


def buildMatrix(data, documents):
    conf = np.zeros((data.C, data.C))  # random vector
    for n in range(data.N):
        topics = documents[n].gold_topics
        c = documents[n].pred_cluster
        for t in topics:
            t_idx = data.topic2index[t]
            conf[c][t_idx] += 1

    conf_df = pd.DataFrame(conf)
    conf_df.columns = data.topic2index.keys()
    cluster_size_df = pd.DataFrame(data.cluster_freq.values())
    cluster_size_df.columns = ['Cluster size']
    mat_df = pd.concat([cluster_size_df, conf_df], axis=1)
    mat_df.index = data.cluster_freq.keys()
    mat_df.index.name = 'Cluster ID'
    mat_df = mat_df.sort_values(by=['Cluster size'], ascending=False)
    mat_df.to_csv('mat_df.csv')
    print(mat_df)
    sns.heatmap(mat_df[data.topic2index.keys()], annot=True, fmt='g')
    plt.title('Confusion Matrix')

    plt.savefig('conf.png')
    plt.clf()
    # plt.show()

    return conf


def buildHistograms(conf, data):
    c2t = {}
    conf_df= pd.DataFrame(conf)
    for c in range(9):
        print("Making histogram for cluster {0}".format(c))
        conf_df.T[c].plot.bar()
        t = (conf_df.T[c]).idxmax()
        c2t[c] = t
        plt.title("Cluster {0} is Topic {1} (topic id: {2})".format(c, data.index2topic[t], t))
        plt.xlabel("Topic ID")

        plt.savefig('hist_{0}.png'.format(c))
        plt.clf()
        # plt.show()

    return c2t


def calcAcc(data, documents, c2t):
    correct = 0
    for n in range(data.N):
        pred_c = documents[n].pred_cluster
        pred_c_t = data.index2topic[c2t[pred_c]]
        gold_topics = documents[n].gold_topics
        if pred_c_t in gold_topics:
            correct += 1

    acc = correct / data.N * 100

    print("Acc is {0:.2f}%".format(acc))


if __name__ == '__main__':
    develop_file_name, topics_file_name = getArgs()
    documents, V_dict = loadDataSet(develop_file_name)
    data = initData(documents, V_dict)
    mStep(data, documents)

    print("[INFO] Starting EM")
    s_time = strftime("%H:%M:%S", gmtime())
    ll_prog = []
    for iter in range(ITERATIONS):
        s_time = strftime("%H:%M:%S", gmtime())
        eStep(data, documents)
        mStep(data, documents)
        ll = calcLL(data, documents)
        ll_prog.append(ll)

        e_time = strftime("%H:%M:%S", gmtime())
        print("\tIter:{0} time: {1} ll: {2}".format(iter, (datetime.strptime(e_time, '%H:%M:%S') - datetime.strptime(s_time, '%H:%M:%S')), ll))

    e_time = strftime("%H:%M:%S", gmtime())
    print("[INFO] [EM] time: {0}".format(datetime.strptime(e_time, '%H:%M:%S') - datetime.strptime(s_time, '%H:%M:%S')))

    print("[INFO] Let's make some statistics")
    makeStats(data, documents, ll_prog)

    # confusion matrix
    setTopicsAndClusters(data, documents, topics_file_name)
    conf = buildMatrix(data, documents)

    # histograms
    c2t = buildHistograms(conf, data)

    # Acc
    calcAcc(data, documents, c2t)




    pass

