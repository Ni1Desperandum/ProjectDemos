#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    synset_list = []
    synlemma_set = set()
    for everylemma in wn.lemmas(str(lemma),str(pos)):
    #Get the lemma's synset
        synset_list.append(everylemma.synset())
    for synset in synset_list:
        for everysynlemma in synset.lemmas():
            if everysynlemma.name() != str(lemma):
                synlemma_set.add(everysynlemma.name())
    list(synlemma_set)
    return synlemma_set
     

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    synonym_freq_dict = dict()
    synonym_names = []
    synonym_frequencies = []
    #Get all of the lemmas
    all_lemmas = wn.lemmas(context.lemma, context.pos)
    for everylemma in all_lemmas:
        #Get the synset of that lemma
        everylemma_synset = everylemma.synset()
        for everylemma_synset_lemma in everylemma_synset.lemmas():
            synonym = everylemma_synset_lemma.name()
            syn_freq = everylemma_synset_lemma.count()
            if synonym != context.lemma and '_' not in synonym:
                synonym_names.append(synonym)
                synonym_frequencies.append(syn_freq)
            elif synonym != context.lemma and '_' in synonym:
                synonym=synonym.replace('_', ' ')
                synonym_names.append(synonym)
                synonym_frequencies.append(syn_freq)
    synonym_freq_dict=dict(zip(synonym_frequencies, synonym_names))
    max_freq_synonym = synonym_freq_dict[max(synonym_frequencies)]
    return max_freq_synonym
    

def wn_simple_lesk_predictor(context : Context) -> str:
    synset = []
    synset_examples_list = []
    synset_hypernyms_list = []
    hypernym_examples_list = []
    hypernym_definitions_list = []
    # Get all stop_words
    stop_words = stopwords.words('english')
    # Get the synsets for the context(target word)
    synset_list = []
    for lemma in wn.lemmas(str(context.lemma),str(context.pos)):
    #Get the lemma's synsets
        synset_list.append(lemma.synset())
    #Look at each of the lemmas sen
    for synset in synset_list:
        #Get all of the examples in the synset
        synset_examples_list.append(synset.examples())
        #Get all of the hypernyms
        synset_hypernyms_list.append(synset.hypernyms())
    for hypernym in synset_hypernyms_list:
        #Get all of the examples for each hypernym
        hypernym_examples_list.append(hypernym.examples())
        #Get all of the definitions for each hypernym
        hypernym_definitions_list.append(hypernym.definitions())        
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        cs_sims =[]
        ordered_syn=[]
        cs_syn_dict =dict()
        #GET ALL THE SYNONYMS
        synonyms = get_candidates(self.context.lemma, self.context.pos)
        #get max of cos similarity as this is the most similar word)
        for syn in synonyms:
            cs_sims.append(self.model.similarity(context.lemma, syn))
            ordered_syn.append(syn)
        cs_syn_dict = dict(zip(cs_sims,ordered_syn))
        max_score_synonym = cs_syn_dict[max(cs_sims)]
        return max_score_synonym


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        best_words = ''
        #Get all of the synonyms
        synonyms = get_candidates(self.context.lemma, context.pos)
        #Create the sentence
        full_sentence=str(context.left_context) + ' [MASK] ' str(context.right_context)
        #Encode the sentence with the masked target word
        encoded_sentence=self.tokenizer.encode(full_sentence)
        tokenized_sentence=self.tokenizer.convert_ids_to_tokens(encoded_sentence)
        masked_target_index=tokenized_sentence.index(context.lemma)
        input_mat = np.array(tokenized_sentence).reshape((1,-1))
        outputs=self.model.predict(input_mat)
        predictions=outputs[0]
        best_words=np.argsort(predictions[0][masked_target_index])[::-1]
        for syn in synonyms:
        if '_' in syn:
          syn = syn.replace("_","")
        if syn in best_words:
          best_word
          break
        else:
          continue

    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    freq_predict=wn_frequency_predictor(context)
    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = smurf_predictor(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
