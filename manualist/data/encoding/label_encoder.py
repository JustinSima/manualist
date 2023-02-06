""" Embed labels as word vectors."""
import re
import os
import json
import numpy as np
import gensim.downloader

from constants import DATA_SOURCE, LABEL_DIRECTORY


GENSIM_MODELS = [
    'word2vec-ruscorpora-300',
    'word2vec-google-news-300',
    'glove-wiki-gigaword-50',
    'glove-wiki-gigaword-100',
    'glove-wiki-gigaword-200',
    'glove-wiki-gigaword-300',
    'glove-twitter-25',
    'glove-twitter-50',
    'glove-twitter-100',
    'glove-twitter-200'
]
PROBLEM_WORDS = {
    "don't":'do not',
    'fingerspell':'finger spell'
}

def map_word(word):
    if word in PROBLEM_WORDS:
        return PROBLEM_WORDS[word]

    else:
        return word

def create_label_embeddings(model_name: str):
    # Load data source.
    with open(DATA_SOURCE, 'r') as f:
        source = json.load(f)

    # Download pretrained model.
    model_vectors = gensim.downloader.load(model_name)
    
    # Class labels as they appear.
    vocab = [word['gloss'] for word in source]

    # Convert each label to a vector.
    word_vectors = []
    for single_label in source:
        word = single_label['gloss']

        # Append regular single words diretly.
        if ' ' not in word and word not in PROBLEM_WORDS:
            word_vectors.append(model_vectors[word])
    
        else:
            # Create phrase embedding by summing individual words.
            phrase = np.empty(shape=(25,))
            separate_words = word.split(' ')

            for new_word in separate_words:
                # Catch problem words here.
                new_word = map_word(new_word)

                # Resplit if the mapped word is actually a subphrase.
                if ' ' not in new_word:
                    phrase += model_vectors[new_word]
                else:
                    for smaller_word in new_word.split(' '):
                        phrase += model_vectors[smaller_word]
            
            word_vectors.append(phrase)

    # Save and return embeddings.
    word_vectors = [embedding.tolist() for embedding in word_vectors]
    embeddings = dict(zip(vocab, word_vectors))

    save_name = os.path.join(LABEL_DIRECTORY, 'label-embeddings-' + model_name + '.json')
    with open(save_name, 'w') as f:
        json.dump(embeddings, f)

    return embeddings
