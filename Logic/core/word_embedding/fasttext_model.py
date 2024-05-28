import fasttext
import re

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance

from .fasttext_data_loader import FastTextDataLoader


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                       punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    
    if lower_case:
        text = text.lower()
    if punctuation_removal:
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
    word_tokens = word_tokenize(text)
    if stopword_removal:
        stopwords_list = set(stopwords.words('english') + stopwords_domain)
        word_tokens = [word for word in word_tokens if word not in stopwords_list]
    word_tokens = [word for word in word_tokens if len(word) >= minimum_length]
    return ' '.join(word_tokens)

class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None


    def train(self, texts):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        import pandas as pd
        columns = ['title',  'summaries', 'synopsis', 'review']
        pd.DataFrame(texts,columns=columns).to_csv('texts.csv')
        self.model = fasttext.train_unsupervised('texts.csv', model=self.method)
    def get_doc_embedding(self, doc):
        ans = None
        tokens = preprocess_text(doc).split()
        for token in tokens:
            if ans is None:
                ans = self.model.get_word_vector(token)
            else:
                ans = [x + y for x, y in zip(ans, self.model.get_word_vector(token))]
        ans =  [x / len(tokens) for x in ans]

        return ans

    def get_query_embedding(self, query):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        ans = None
        
        for token in preprocess_text(query).split():
            if ans is None:
                ans = self.model.get_word_vector(token)
            else:
                ans = [x + y for x, y in zip(ans, self.model.get_word_vector(token))]
        ans =  [x / len(preprocess_text(query).split()) for x in ans]

        return ans
    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        word1_vec = self.model.get_word_vector(word1)
        word2_vec = self.model.get_word_vector(word2)
        word3_vec = self.model.get_word_vector(word3)

        # Perform vector arithmetic
        ans = word2_vec - word1_vec + word3_vec

        # Create a dictionary mapping each word in the vocabulary to its corresponding vector
        word_vectors = {}
        for word in self.model.words:
            word_vectors[word] = self.model.get_word_vector(word)
        print(len(word_vectors))

        # Exclude the input words from the possible results
        word_vectors.pop(word1)
        word_vectors.pop(word2)
        word_vectors.pop(word3)

        # Find the word whose vector is closest to the result vector
        min_dist = float('inf')
        closest = None
        for word, vec in word_vectors.items():
            
            dist = sum((vec - ans) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest = word

        return closest

    def save_model(self, path='FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)

    def load_model(self, path="FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)

    def prepare(self, dataset, mode, save=False, path='FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_model(path)
        if mode == 'save':
            self.save_model(path)

if __name__ == "__main__":
    inp = input('train?')
    if inp =='yes':
        ft_model = FastText(method='skipgram')

        path = 'IMDB_crawled.json'
        ft_data_loader = FastTextDataLoader(preprocessor=preprocess_text,file_path=path)

        X,y = ft_data_loader.create_train_data()
        # print('*' * 20)

        ft_model.train(X)
        ft_model.prepare(None, mode = "save")
    else:
        ft_model = FastText(method='skipgram')
        ft_model.prepare(None, mode = "load")
    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'war'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "girl"
    word2 = "mother"
    word3 = "boy"
    print(f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
