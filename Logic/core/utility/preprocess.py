from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import string

class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        self.documents = documents

        path = './Logic/core/utility/stopwords.txt'
        with open(path, "r") as file:
            self.stopwords = file.readlines()
        for i in range(len(self.stopwords)):
            self.stopwords[i] = self.stopwords[i].strip()

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        ans = []
        for text in self.documents:
            text = self.normalize(text)
            text = self.remove_links(text)
            text = self.remove_punctuations(text)
            text = self.remove_stopwords(text)
            text = ' '.join(text)
            ans.append(text)

        return ans

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        text = text.lower()
            # Perform stemming
        words = self.tokenize(text)

        
        # words = [self.stemmer.stem(word) for word in words]


        # words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        for pat in patterns:
            text = re.sub(pat, '', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """

        text = text.replace('\n',' ').replace('  ',' ')
        ans = text.split(' ')
        
        return [i for i in ans if i!='']

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        words = self.tokenize(text)
        return [word for word in words if word not in self.stopwords]

# a = Preprocessor(['it is www.a.com? but are you ok!!! or not?i am not\nok    in this '])
# print(a.normalize('hello how can i help you he is back and rides a horse with operational data'))
# print(a.remove_links('it is ok in www.goog.com or http://hell.ar or amir.org'))
# print(a.preprocess())
# print(a.tokenize('i am not\nok    in this '))

def preprocess_docs(docs):
    
    from indexer.indexes_enum import Indexes
    fields = (Indexes.GENRES.value,Indexes.SUMMARIES.value,Indexes.STARS.value)
    
    for field in fields:
        for doc in docs:
            preprocessor = Preprocessor(doc[field])
            doc[field] = preprocessor.preprocess()

if __name__ == '__main__':
    import json
    json_file_path = './IMDB_crawled.json'
    with open(json_file_path, "r") as file:
        docs = json.load(file)

    preprocess_docs(docs)
    with open('./IMDB_preprocessed.json', "w") as file:
        file.write(json.dumps(docs))

