import sys


DEV = True
### Globals start ###
OUTPUTS = {}  # global dict of outputs
V = 300000  # given vocab size
### Globals end ###
S = 0
dev_tokens = []
test_tokens = []

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
    impl. init part (specifically (e) and (f))of the pdf
    """
    setOutputInfo("Output5", V)  # (e), 300,000

    p_uniform = 1/V
    setOutputInfo("Output6", p_uniform)  # (f), uniform distribution


def events2Dict(set_file_name, tokens):
    """
    scan the articles txt, every word that appears in the article, push into the dict and increase it's counter by 1.
    """

    d = dict()
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
    global dev_tokens
    global test_tokens
    dev_events, dev_tokens = events2Dict(dev_set_file_name, dev_tokens)
    test_events, test_tokens = events2Dict(test_set_file_name, test_tokens)

    return dev_events, test_events



def printOutput(output_file_name):
    with open(output_file_name, 'w') as f:
        for key, val in OUTPUTS.items():
            out_print = "#" + key + '\t' + str(val) + '\n'
            f.write(out_print)
            if DEV:
                print(out_print, end="")


def devSetPreProcessing(dev_events):
    """
    impl. Development set preprocessing part of the pdf
    iterate the events dict, summerise the keys instances
    """
    global S
    S = sum([val for key, val in dev_events.items()])

    setOutputInfo("Output7", S)

def createTrainValSet(dev_events, input_word):
    """
        Split dev into train (90%) and val(10%) sets of the total words (not events).
        Meaning we count also the number of times each word appears until reaching 90%.
    """
    train = dev_tokens[0: round(0.90*S)]
    val = dev_tokens[round(0.90*S):]
    train_set = set(train)
    val_set = set(val)

    input_word_freq = dev_events[input_word] if input_word in dev_events else 0
    setOutputInfo("Output8", len(val)) #I know in the pdf he says events, but then output 10 would be the same as 9 :/
    setOutputInfo("Output9", len(train))
    setOutputInfo("Output10", len(train_set))
    setOutputInfo("Output11", input_word_freq)

    return train_set, val_set





if __name__ == '__main__':
    dev_set_file_name, test_set_file_name, input_word, output_file_name = getArgs()
    init()
    dev_events, test_events = loadSets(dev_set_file_name, test_set_file_name)
    devSetPreProcessing(dev_events)
    train_set, val_set = createTrainValSet(dev_events, input_word)

    printOutput(output_file_name)


