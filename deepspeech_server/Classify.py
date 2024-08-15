''' Module for classifying sentences to assist with the results of transcriptions '''

from sentence_transformers import SentenceTransformer, util
import numpy as np
import csv

BERTIMBAU_MODEL = 'neuralmind/bert-base-portuguese-cased'

class Classify:

    def __init__(self, model, path_phrases):
        self.model = self.__load_model(model) # Load model
        self.phrases = self.__load_phrases(path_phrases)  # Generate list of phrases from csv file
        self.phrases_tensors = self.__generate_phrases_tensor(self.phrases)

    def __load_model(self, model_name):
        ''' Load model STS from Hugging Face '''
        return SentenceTransformer(model_name)


    def __load_phrases(self, path):
        ''' Load phrases from csv file '''
        phrases = []
        try:
            with open(path, encoding='utf-8', newline='') as senteces_csv:  # No header
                reader_csv = csv.reader(senteces_csv, delimiter=',')
                for phrase in reader_csv:
                    phrases.append(phrase[0])
        except Exception as err: 
            print(f'Error: {err}')
        finally:
            return phrases


    def __generate_phrases_tensor(self, phrases):
        ''' Generate tensor from phrases '''
        tensors = self.model.encode(phrases, convert_to_tensor=True)
        print(f'LOG --- Quantidade de frases-tensores: {len(tensors)}')
        return tensors 
    

    def generate_tensor(self, phrase):
        ''' Generate tensor from phrase (string) '''
        return self.model.encode(phrase, convert_to_tensor=True)


    def find_more_similar(self, phrase):
        ''' Find the sentence with the greatest similarity within the set '''
        phrase_tensor = self.generate_tensor(phrase)        # Tensor for the phrase
        similarities = util.pytorch_cos_sim(phrase_tensor, self.phrases_tensors)
        idx_most_similar = np.argmax(similarities)
        
        sentence = self.phrases[idx_most_similar]           # Most similar sentence
        score = float(similarities[0][idx_most_similar])    # Similarity score

        return sentence, score