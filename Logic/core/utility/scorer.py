import numpy as np


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.

        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.

        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.

        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            if term not in self.index:
                idf = 0
            else:
                df = len(self.index[term])
                if df == 0:
                    idf = 0
                else:
                    idf = np.log2(self.N / df)

            self.idf[term] = idf

        return idf

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        
        ans = {}
        for term in query:
            if term in ans:
                ans[term] += 1
            else:
                ans[term] = 1

        return ans

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        ans ={}

        q_tf = self.get_query_tfs(query)
        docs = self.get_list_of_documents(query)

        for doc in docs:
            ans[doc] = self.get_vector_space_model_score(query, q_tf, doc, method[:3], method[4:])
        # print(ans)
        return ans

    def get_vector_space_model_score(
        self, query, query_tfs, document_id, document_method, query_method
    ):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """
        query_w = []
        doc_w = []
        terms = list(set(query))

        terms = [term for term in terms if term in self.index]

        for term in terms:
            
            q_tf  = query_tfs[term]
            if query_method[0] == 'l':
                q_tf = 1 + np.log10(q_tf)
            q_idf = 1
            if query_method[1] == 't':
                q_idf = self.get_idf(term)
            query_w.append(q_idf*q_tf)

            d_tf = self.index[term].get(document_id,0)
            if document_method[0] == 'l':
                if d_tf != 0:
                    d_tf = 1 + np.log10(d_tf)
            d_idf = 1
            if document_method[1] == 't':
                d_idf = self.get_idf(term)
            doc_w.append(d_idf*d_tf)

        


        if query_method[2] == 'c':
            
            total = 0
            for weight in query_w:
                total += weight ** 2
            query_normalization = np.sqrt(total)
            query_w = [weight / query_normalization for weight in query_w]
        
        
        if document_method[2] == 'c':
            
            total = 0
            for weight in doc_w:
                total += weight ** 2
            doc_normalization = np.sqrt(total)
            if doc_normalization !=0:
                doc_w = [weight / doc_normalization for weight in doc_w]

        final_score = sum([doc_weight * query_weight for (query_weight, doc_weight) in zip(query_w, doc_w)])
        return final_score

    def compute_socres_with_okapi_bm25(
        self, query, average_document_field_length, document_lengths
    ):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        documents = self.get_list_of_documents(query)
        ans = {}
        for document in documents:
            ans[document] = self.get_okapi_bm25_score(query, document, average_document_field_length, document_lengths)
        return ans

    def get_okapi_bm25_score(
        self, query, document_id, average_document_field_length, document_lengths
    ):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """
        terms = list(set(query))
        terms = [term for term in terms if term in self.index]
        ans = 0
        k = 2
        b = 0.75

        for term in terms:
            tf = self.index[term].get(document_id,0)
            idf = self.get_idf(term)
            document_length = document_lengths[document_id]

            other = ((k + 1) * tf) / (k * ((1 - b) + b * (document_length / average_document_field_length)) + tf)
            score = idf * other
            ans += score

        return ans


    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            A dictionary of the document IDs and their scores.
        """

        # TODO
        pass

    def compute_score_with_unigram_model(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        document_id : str
            The document to calculate the score for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            The Unigram score of the document for the query.
        """

        # TODO
        pass

