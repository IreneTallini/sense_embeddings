import os

import tensorflow as tf
import numpy as np
import tqdm
from tensorboard.plugins import projector
from data_preprocessing import split_in_batches, generate_batch, build_dataset, save_vectors, read_analogies
from evaluation import evaluation

import random

# run on CPU
# comment this part if you want to run it on GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

### PARAMETERS ###

BATCH_SIZE = 60 #Number of samples per batch
EMBEDDING_SIZE = 150 # Dimension of the embedding vector.
WINDOW_SIZE = 5  # How many words to consider left and right.
NEG_SAMPLES = 5  # Number of negative examples to sample.
VOCABULARY_SIZE = 3000 #The most N word to consider in the dictionary

TRAIN_DIR = "dataset/DATA/TRAIN"
VALID_DIR = "dataset/DATA/DEV"
TMP_DIR = "/tmp/"
ANALOGIES_FILE = "dataset/eval/questions-words.txt"


### READ THE TEXT FILES ###
#Read the data into a list of sentences (lists of words). 
def read_data(directory, domain_lines=-1):
    data = []
    for domain in os.listdir(directory):
        limit = domain_lines
        
        #Next three variables are used to avoid reading all data from the same file 
        files_number = len(os.listdir(os.path.join(directory, domain)))
        quotient, remainder = divmod(domain_lines, files_number)
        number_of_lines_list = [quotient + 1] * remainder + [quotient] * (files_number - remainder)
        
        for f_counter, f in enumerate(os.listdir(os.path.join(directory, domain))):
            lines_per_file = number_of_lines_list[f_counter]
            if lines_per_file <= 0:
                break
            if f.endswith(".txt"):
                with open(os.path.join(directory, domain, f), encoding='utf-8') as file:
                    for line in file.readlines():
                        split = line.lower().strip().split()
                        data += [split]
                        lines_per_file -= 1
                        if lines_per_file <= 0:
                            break
    return data

# load the training set
raw_data = read_data(TRAIN_DIR, domain_lines=10000)
print('Data size', len(raw_data))
# the portion of the training set used for data evaluation
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


### CREATE THE DATASET AND WORD-INT MAPPING ###

data, dictionary, reverse_dictionary = build_dataset(raw_data, VOCABULARY_SIZE)
del raw_data  # Hint to reduce memory.
# read the question file for the Analogical Reasoning evaluation
questions = read_analogies(ANALOGIES_FILE, dictionary)

### DELETE FREQUENT WORDS ###
#This function deletes words w with frequence f(w) with prob given by f(w)^(1/4)
def delete_frequent_words(data, reverse_dictionary):
    word_counts = {word : 1 for word in reverse_dictionary}
    for line in data:
        for word in line:
            word_counts[word] += 1
    total_count = sum(word_counts)
    freqs = {word: count/total_count for word, count in word_counts.items()}
    p_drop = {word : freqs[word]**(1/4) for word in reverse_dictionary}
    return [[word for word in line if random.random() < 1 - p_drop[word]] for line in data]

data = delete_frequent_words(data, reverse_dictionary)

### MODEL DEFINITION ###

graph = tf.Graph()
eval = None

with graph.as_default():
    # Define input data tensors.
    with tf.name_scope('inputs'):
        train_input = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        train_label = tf.placeholder(tf.int64, shape=[BATCH_SIZE, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        
    # Lookup Layer
    with tf.name_scope('lookup_table'):   
        embeddings = tf.Variable(tf.random_normal([VOCABULARY_SIZE, EMBEDDING_SIZE]))
        output_layer_1 = tf.nn.embedding_lookup(embeddings, train_input)
        
    #Softmax layer
    with tf.name_scope('output'):    
        softmax = tf.Variable(tf.random_normal([VOCABULARY_SIZE, EMBEDDING_SIZE]))
        softmax_b = tf.Variable(tf.zeros(VOCABULARY_SIZE))
    
    #Loss implementing negative sampling
    with tf.name_scope('loss'):
        cost = tf.nn.sampled_softmax_loss(
            weights=softmax,
            biases=softmax_b,
            labels=train_label,
            inputs=output_layer_1,
            num_sampled=NEG_SAMPLES,
            num_classes=VOCABULARY_SIZE)
        loss = tf.reduce_mean(cost)
                                                              
    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)
    
    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    
    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()
    
    # evaluation graph
    eval = evaluation(normalized_embeddings, dictionary, questions)


### TRAINING ###

# Step 5: Begin training.
num_steps = 1000000

with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(TMP_DIR, session.graph)
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    bar = tqdm.tqdm(range(num_steps))
    # Split data in batches
    batches = split_in_batches(BATCH_SIZE, WINDOW_SIZE, data)
    
    for step in bar:
        batch_inputs, batch_labels = generate_batch(step, batches)
        batch_inputs = np.array(batch_inputs)
        batch_labels = np.array(batch_labels)

        # Define metadata variable.
        run_metadata = tf.RunMetadata()

        # We perform one update step by evaluating the optimizer op
        _, summary, loss_val = session.run(
            [optimizer, merged, loss],
            feed_dict={train_input: batch_inputs, train_label: batch_labels},
            run_metadata=run_metadata)
        average_loss += loss_val

        # Add returned summaries to writer in each step.
        writer.add_summary(summary, step)
        # Add metadata to visualize the graph for the last run.
        if step % 1000 == 0 and step != 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step)
        if step % 1000 == 0 and step != 0:
            #Add a summary for accuracy
            summary_accuracy = tf.Summary()
            summary_accuracy.value.add(tag="accuracy", simple_value=eval.eval(session))
            writer.add_summary(summary_accuracy, step)
            print("avg loss: "+str(average_loss/step))
    final_embeddings = normalized_embeddings.eval()
    
    

    ### SAVE VECTORS ###

    save_vectors(final_embeddings, os.path.join(TMP_DIR, 'final_embeddings'), dictionary)

    # Write corresponding labels for the embeddings.
    with open(TMP_DIR + '/metadata.tsv', 'w', encoding='utf-8') as f:
        for i in range(VOCABULARY_SIZE):
            f.write(reverse_dictionary[i] + '\n')

    # Save the model for checkpoints
    saver.save(session, os.path.join(TMP_DIR, 'model.ckpt'))

    # Create a configuration for visualizing embeddings with the labels in TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(writer, config)

writer.close()
