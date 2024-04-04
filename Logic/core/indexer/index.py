import time
import os
import json
import copy
from indexes_enum import Indexes


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        for doc in self.preprocessed_documents:
            current_index[doc['id']] = doc

        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        index = {}
        for doc in self.preprocessed_documents:
            doc_id = doc['id']
            actors = doc['stars']
            count = 0
            for actor in actors:
                count += len(actor.split())
            for actor in actors:
                
                for term in actor.split():
                    
                    if term not in index:
                        index[term] = {}
                    if doc_id not in index[term]:
                        index[term][doc_id] = 0
                    index[term][doc_id] += 1/count
            
            
        return index
        

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        index = {}
        for doc in self.preprocessed_documents:
            doc_id = doc['id']
            actors = doc['genres']
            count = 0
            for actor in actors:
                count += len(actor.split())
            for actor in actors:
                
                for term in actor.split():
                    
                    if term not in index:
                        index[term] = {}
                    if doc_id not in index[term]:
                        index[term][doc_id] = 0
                    index[term][doc_id] += 1/count
            
            
        return index        

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """


        index = {}
        for doc in self.preprocessed_documents:
            doc_id = doc['id']
            actors = doc['summaries']
            count = 0
            for actor in actors:
                count += len(actor.split())
            for actor in actors:
                
                for term in actor.split():
                    
                    if term not in index:
                        index[term] = {}
                    if doc_id not in index[term]:
                        index[term][doc_id] = 0
                    index[term][doc_id] += 1/count
            
            
        return index
    

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            dic = self.index[index_type][word]
            if index_type==Indexes.DOCUMENTS.value:
                return [dic]
            
            ans = []
            for i in dic:
                ans.append(i)
            return ans
        except:

            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """
        self.index[Indexes.DOCUMENTS.value][document['id']] = document

        fields = (Indexes.GENRES.value,Indexes.SUMMARIES.value,Indexes.STARS.value)
        
        for field in fields:
            ind = self.index[field]

            doc_id = document['id']
            actors = document[field]
            count = 0
            for actor in actors:
                count += len(actor.split())
            for actor in actors:
                
                for term in actor.split():
                    
                    if term not in ind:
                        ind[term] = {}
                    if doc_id not in ind[term]:
                        ind[term][doc_id] = 0
                    ind[term][doc_id] += 1/count



    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        document = self.index[Indexes.DOCUMENTS.value].pop(document_id, None)
        if document is None:
            return
        
        fields = (Indexes.GENRES.value,Indexes.SUMMARIES.value,Indexes.STARS.value)

        for field in fields:
            index = self.index[field]
            
            actors = document[field]
            terms = set()
            for actor in actors:
                terms.update(actor.split())
            for term in terms:
                index[term].pop(document_id) # not use None to raise error because it should exist
            
        


    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(set(index_before_add[Indexes.STARS.value]['henry']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str = './indexes/', index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name not in self.index:
            raise ValueError('Invalid index name')

        # TODO
        pass

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """

        with open(path, 'r') as f:
            return json.load(f)




    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # TODO: based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time <= brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

#  Run the class with needed parameters, then run check methods and finally report the results of check methods




def main():
    json_file_path = './IMDB_preprocessed.json'


    with open(json_file_path, "r") as file:
        docs = json.load(file)
    

    # print(len(docs))
    index = Index(docs)
    print('created index')
    index.store_index(index_type=Indexes.GENRES.value)
    index.store_index(index_type=Indexes.DOCUMENTS.value)
    index.store_index(index_type=Indexes.SUMMARIES.value)
    index.store_index(index_type=Indexes.STARS.value)

    index.check_add_remove_is_correct()

    loaded = index.load_index('./indexes/' + Indexes.GENRES.value + '_index.json')

    print('index loaded correctly:',index.check_if_index_loaded_correctly(Indexes.GENRES.value, loaded)==True)
    print('test doc index:')
    index.check_if_indexing_is_good(index_type=Indexes.DOCUMENTS, check_word='tt0050083')
    index.check_if_indexing_is_good(index_type=Indexes.DOCUMENTS, check_word='tt0071562')
    print('test star index:')
    index.check_if_indexing_is_good(index_type=Indexes.STARS, check_word='cooper')
    index.check_if_indexing_is_good(index_type=Indexes.STARS, check_word='chevalier')
    print('test genre index:')
    index.check_if_indexing_is_good(index_type=Indexes.GENRES, check_word='drama')
    index.check_if_indexing_is_good(index_type=Indexes.GENRES, check_word='horror')
    print('test summeries index:')
    index.check_if_indexing_is_good(index_type=Indexes.SUMMARIES, check_word='corleone')
    index.check_if_indexing_is_good(index_type=Indexes.SUMMARIES, check_word='flannagan')
    

if __name__ == '__main__':
    main()

