import sys


DEV = True
### Globals start ###
OUTPUTS = {}  # global dict of outputs
V = 300000  # given vocab size
### Globals end ###

class Data:
    def __init__(self):
        self.S = 0
        self.dev_tokens = 0
        self.test_tokens = 0
        self.train_set = set()
        self.val_set = set()
        self.train = []
        self.val = []
        self.dev_events = {}
        self.test_events = {}

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
    data.dev_events, data.dev_tokens = events2Dict(dev_set_file_name, [])
    data.test_events, data.val_tokens = events2Dict(test_set_file_name, [])




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

def createTrainValSet(input_word):
    """
        Split dev into train (90%) and val(10%) sets of the total words (not events).
        Meaning we count also the number of times each word appears until reaching 90%.
    """
    data.train = data.dev_tokens[0: round(0.90*data.S)]
    data.val = data.dev_tokens[round(0.90*data.S):]
    data.train_set = set(data.train)
    data.val_set = set(data.val)

    input_word_freq = data.dev_events[input_word] if input_word in data.dev_events else 0
    setOutputInfo("Output8", len(data.val)) #I know in the pdf he says events, but then output 10 would be the same as 9 :/
    setOutputInfo("Output9", len(data.train))
    setOutputInfo("Output10", len(data.train_set))
    setOutputInfo("Output11", input_word_freq)
    setOutputInfo("Output12", input_word_freq / len(data.train))
    setOutputInfo("Output13", 0 / len(data.train))






if __name__ == '__main__':
    dev_set_file_name, test_set_file_name, input_word, output_file_name = getArgs() #1-4
    init() #5-6
    loadSets(dev_set_file_name, test_set_file_name)
    devSetPreProcessing() #7
    createTrainValSet(input_word) #8-13

    printOutput(output_file_name)


