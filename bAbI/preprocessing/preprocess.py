import sys
import pickle
import getopt
import random
import numpy as np
from shutil import rmtree
from os import listdir, mkdir
from os.path import join, isfile, isdir, dirname, basename, normpath, abspath, exists
from argparse import ArgumentParser


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def create_dictionary(files_list):
    """
    creates a dictionary of unique lexicons in the dataset and their mapping to numbers

    Parameters:
    ----------
    files_list: list
        the list of files to scan through

    Returns: dict
        the constructed dictionary of lexicons
    """

    lexicons_dict = {}
    id_counter = 0

    llprint("Creating Dictionary ... 0/%d" % (len(files_list)))

    for indx, filename in enumerate(files_list):
        with open(filename, 'r') as fobj:
            for line in fobj:

                # first seperate . and ? away from words into seperate lexicons
                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')

                for word in line.split():
                    if not word.lower() in lexicons_dict and word.isalpha():
                        # 0 is reserved for padding
                        id_counter += 1
                        lexicons_dict[word.lower()] = id_counter

        llprint("\rCreating Dictionary ... %d/%d" % ((indx + 1), len(files_list)))

    print("\rCreating Dictionary ... Done!")
    return lexicons_dict


def encode_data(files_list, lexicons_dictionary, commun_length, max_q, length_limit=None):
    """
    encodes the dataset into its numeric form given a constructed dictionary

    Parameters:
    ----------
    files_list: list
        the list of files to scan through
    lexicons_dictionary: dict
        the mappings of unique lexicons

    Returns: tuple (dict, int)
        the data in its numeric form, maximum story length
    """

    files = {}
    story_inputs = None
    story_outputs = None
    answers = []
    answers_sub = []
    length_answers = []
    stories_lengths = []
    answers_flag = False  # a flag to specify when to put data into outputs list
    limit = length_limit if not length_limit is None else float("inf")
    llprint("Encoding Data ... 0/%d" % (len(files_list)))

    for indx, filename in enumerate(files_list):

        files[filename] = []
        task = filename.split("/")[-1].split("_")[0]

        with open(filename, 'r') as fobj:
            for line in fobj:

                # first seperate . and ? away from words into seperate lexicons
                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')

                answers_flag = False  # reset as answers end by end of line

                for i, word in enumerate(line.split()):

                    if word == '1' and i == 0:
                        # beginning of a new story
                        if not story_inputs is None:
                            stories_lengths.append(len(story_inputs))
                            if len(answers_sub) > 0:
                                answers.append(answers_sub)
                                length_answers.append(len(answers_sub))
                            assert sum(length_answers) == len(story_outputs)
                            length_answers += (max_q - len(length_answers)) * [-1]
                            answers_sub = []
                            if len(story_inputs) <= limit:
                                # pad the input and output if necessary

                                if len(story_inputs) < commun_length:
                                    story_inputs += (commun_length - len(story_inputs)) * [lexicon_dictionary["*"]]
                                else:
                                    story_inputs = story_inputs[:commun_length]
                                files[filename].append({
                                    'inputs': story_inputs,
                                    'outputs': story_outputs,
                                    'length_answers': length_answers,
                                    'task': task
                                })

                        story_inputs = []
                        story_outputs = []
                        answers = []
                        length_answers = []

                    if word.isalpha() or word == '?' or word == '.':
                        if not answers_flag:
                            story_inputs.append(lexicons_dictionary[word.lower()])
                        else:
                            story_inputs.append(lexicons_dictionary['-'])
                            story_outputs.append(lexicons_dictionary[word.lower()])
                            answers_sub.append(lexicons_dictionary[word.lower()])
                        # set the answers_flags if a question mark is encountered
                        if not answers_flag:
                            answers_flag = (word == '?')
                            if len(answers_sub) > 0:
                                answers.append(answers_sub)
                                length_answers.append(len(answers_sub))
                                answers_sub = []

        llprint("\rEncoding Data ... %d/%d" % (indx + 1, len(files_list)))

    print("\rEncoding Data ... Done!")
    return files, stories_lengths


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-d",
                            "--data_dir",
                            required=True,
                            help="Path to the data directory.")

    arg_parser.add_argument("-t",
                            "--task_dir",
                            required=True,
                            help="The task directory.")

    arg_parser.add_argument("-l",
                            "--length_limit",
                            required=True,
                            help="Lenght limit.",
                            type=int)

    arg_parser.add_argument("-c",
                            "--commun_length",
                            required=True,
                            help="commun story length.",
                            type=int)

    arg_parser.add_argument("-q",
                            "--max_q",
                            required=True,
                            help="Max number of questions",
                            type=int)

    arg_parser.add_argument("-m",
                            "--mode",
                            required=True,
                            help="single or joint train.")

    args = vars(arg_parser.parse_args())
    data_dir = args["data_dir"]
    task_dir = args["task_dir"]
    if args["mode"] == "single":
        joint_train = False
    elif args["mode"] == "joint":
        joint_train = True
    else:
        raise NotImplementedError("Choose between <single> and <joint>")
    length_limit = args["length_limit"]
    files_list = []
    commun_length = args["commun_length"]
    max_q = args["max_q"]
    for entryname in listdir(data_dir):
        entry_path = join(data_dir, entryname)
        if isfile(entry_path):
            files_list.append(entry_path)

    lexicon_dictionary = create_dictionary(files_list)
    lexicon_count = len(lexicon_dictionary)

    # append used punctuation to dictionary
    lexicon_dictionary['?'] = lexicon_count + 1
    lexicon_dictionary['.'] = lexicon_count + 2
    lexicon_dictionary['-'] = lexicon_count + 3
    lexicon_dictionary["*"] = 0  # used for padding

    idx_to_word = {v: k for (k, v) in lexicon_dictionary.items()}
    encoded_files, stories_lengths = encode_data(files_list, lexicon_dictionary, commun_length, max_q, length_limit)

    stories_lengths = np.array(stories_lengths)
    length_limit = np.max(stories_lengths) if length_limit is None else length_limit
    print("Total Number of stories: %d" % (len(stories_lengths)))
    print("Number of stories with lengthes > %d: %d (%% %.2f) [discarded]" % (length_limit, np.sum(stories_lengths > length_limit), np.mean(stories_lengths > length_limit) * 100.0))
    print("Number of Remaining Stories: %d" % (len(stories_lengths[stories_lengths <= length_limit])))
    keys = list(encoded_files.keys())
    for k in keys:
        if "train" in k:
            data = encoded_files[k]
            length_val = int(0.1 * len(data))
            random.shuffle(data)
            k_val = k.replace("train", "val")
            encoded_files[k_val] = data[:length_val]
            encoded_files[k] = data[length_val:]

    processed_data_dir = join(task_dir, 'data', basename(normpath(data_dir)))
    train_data_dir = join(processed_data_dir, 'train')
    val_data_dir = join(processed_data_dir, 'val')
    test_data_dir = join(processed_data_dir, 'test')
    if exists(processed_data_dir) and isdir(processed_data_dir):
        rmtree(processed_data_dir)

    mkdir(processed_data_dir)
    mkdir(train_data_dir)
    mkdir(val_data_dir)
    mkdir(test_data_dir)

    llprint("Saving processed data to disk ... ")

    pickle.dump(lexicon_dictionary, open(join(processed_data_dir, 'lexicon-dict.pkl'), 'wb'))
    pickle.dump(idx_to_word, open(join(processed_data_dir, 'idx_to_word.pkl'), 'wb'))

    joint_train_data = []
    joint_val_data = []

    for filename in encoded_files:
        if filename.endswith("test.txt"):
            pickle.dump(encoded_files[filename], open(join(test_data_dir, basename(filename) + '.pkl'), 'wb'))
        elif filename.endswith("train.txt"):
            if not joint_train:
                pickle.dump(encoded_files[filename], open(join(train_data_dir, basename(filename) + '.pkl'), 'wb'))
            else:
                joint_train_data.extend(encoded_files[filename])
        elif filename.endswith("val.txt"):
            if not joint_train:
                pickle.dump(encoded_files[filename], open(join(val_data_dir, basename(filename) + '.pkl'), 'wb'))
            else:
                joint_val_data.extend(encoded_files[filename])

    if joint_train:
        pickle.dump(joint_train_data, open(join(train_data_dir, 'train.pkl'), 'wb'))
        pickle.dump(joint_val_data, open(join(val_data_dir, 'val.pkl'), 'wb'))

    llprint("Done!\n")
