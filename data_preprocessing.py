import collections
import string

import numpy as np


### split_in_batches ###
# This function splits the dataset in batches of equal size

### Parameters ###
# batch_size: the number of train_data,label pairs to produce per batch
# window_size: the size of the context
# data: the dataset

### Return values ###
# data_split: list of batches (lists) of (central_word, neighbor_word) tuples.
# For every central word there are 2*WINDOW_SIZE neighbors words
 
def split_in_batches(batch_size, WINDOW_SIZE, data):
 
    ### neighbors ###
    #auxiliary function which takes as input a list of sentences (list of lists of strings)
    #and outputs a list of tuples (central_word, neighbor_word). 
    
    def neighbors(lines, WINDOW_SIZE):
        data = []
        for line in lines:
            for index, central_word in enumerate(line):
                for neighbor_word in line[max(index - WINDOW_SIZE, 0) : min(index + WINDOW_SIZE + 1, len(line))]:
                    if neighbor_word != central_word :
                        data.append((central_word, neighbor_word))
        return data

    data = neighbors(data, WINDOW_SIZE)
    data_split = [data[i*batch_size : i*batch_size+batch_size] for i in range(int(len(data)/batch_size))]            
    return data_split

### generate_batch ###
# This function simply selects the current batch
    
### Parameters ###
#curr_batch: an int which indicates the batch to select
#data: list of sentences already split in batches
    
### Return values ###
#Train_data: list of central words
#labels: list of corresponding neighbor words
def generate_batch(curr_batch, data):
    batch = data[curr_batch % len(data)]
    train_data = [tupla[0] for tupla in batch]
    labels = [[tupla[1]] for tupla in batch]

    return train_data, labels


### build_dataset ###
# This function is responsible of generating the dataset and dictionaries.
# While constructing the dictionary take into account the unseen words by
# retaining the rare (less frequent) words of the dataset from the dictionary
# and assigning to them a special token in the dictionary: UNK. This
# will train the model to handle the unseen words.
# This function also deletes stop words and characters not in [a,...,z].

### Parameters ###
# lines: a list of sentences
# vocab_size:  the size of vocabulary
#
### Return values ###
# data: list of codes (integers from 0 to vocabulary_size-1).
#       This is the original text but words are replaced by their codes
# dictionary: map of words(strings) to their codes(integers)
# reverse_dictionary: maps codes(integers) to words(strings)

def build_dataset(lines, vocab_size):
    data = []
    
    stop_words = open('stop_words.txt', 'r').read().split('\n')[1:]
    ascii_chars = {chr(i) for i in range(97, 123)}
    #The following line deletes all the stop-words and words which contain chars 
    # outside [a,...,z]
    lines_temp = [list(filter(lambda word : 
                set([char for char in word]) & ascii_chars == set([char for char in word]) 
                and not word in stop_words, line)) 
                for line in lines]
            
    # count_dictionary maps words to their count in lines
    count_dictionary = {word : 0 for line in lines_temp for word in line}
    for line in lines_temp:
        for word in line:
            count_dictionary[word] += 1
    
    #keys are sorted by their count (descending) and just the vocab_size more frequent 
    #are kept
    sorted_keys = ['UNK'] + sorted(count_dictionary, key = count_dictionary.get)[::-1]                
    dictionary = {sorted_keys[i] : i for i in range(min(vocab_size, len(sorted_keys)))}
    reversed_dictionary = {item[1] : item[0] for item in list(dictionary.items())}
    
    # words are replaced by their int mapping in the original sentences
    keys_no_UNK = list(dictionary.keys())[1:]
    for line in lines_temp:
        data.append(list(map(lambda word : 
                    dictionary['UNK'] if word not in keys_no_UNK else dictionary[word], 
                    line)))
        
    return data, dictionary, reversed_dictionary

### Save vectors ###
# Save embedding vectors in a suitable representation for the domain identification task
###
def save_vectors(vectors, file, dictionary):
    d = {word : embedding for word, embedding in zip(dictionary.keys(), vectors)}
    np.save(file, d)


# Reads through the analogy question file.
#    Returns:
#      questions: a [n, 4] numpy array containing the analogy question's
#                 word ids.
#      questions_skipped: questions skipped due to unknown words.
#
def read_analogies(file, dictionary):
    questions = []
    questions_skipped = 0
    with open(file, "r") as analogy_f:
        for line in analogy_f:
            if line.startswith(":"):  # Skip comments.
                continue
            words = line.strip().lower().split(" ")
            ids = [dictionary.get(str(w.strip())) for w in words]
            if None in ids or len(ids) != 4:
                questions_skipped += 1
            else:
                questions.append(np.array(ids))
    print("Eval analogy file: ", file)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    return np.array(questions, dtype=np.int32)
