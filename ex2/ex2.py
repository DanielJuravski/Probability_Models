import sys


DEV = True
### Globals start ###
OUTPUTS = {}  # global dict of outputs
V = 300000  # given vocab size
### Globals end ###


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


def events2Dict(set_file_name):
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
                    if word in d:
                        d[word] += 1
                    else:
                        d[word] = 1

    return d


def loadSets(dev_set_file_name, test_set_file_name):
    """
    Load the data sets files into event objects (dicts)
    """
    dev_events = events2Dict(dev_set_file_name)
    test_events = events2Dict(test_set_file_name)

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
    events_number = 0
    for key, val in dev_events.items():
        events_number += val

    setOutputInfo("Output7", events_number)



if __name__ == '__main__':
    dev_set_file_name, test_set_file_name, input_word, output_file_name = getArgs()
    init()
    dev_events, test_events = loadSets(dev_set_file_name, test_set_file_name)
    devSetPreProcessing(dev_events)

    printOutput(output_file_name)


