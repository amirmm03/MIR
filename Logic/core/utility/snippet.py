class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

        path = './Logic/core/utility/stopwords.txt'
        with open(path, "r") as file:
            self.stopwords = file.readlines()
        for i in range(len(self.stopwords)):
            self.stopwords[i] = self.stopwords[i].strip()

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        for stopword in self.stopwords:
            query = query.replace(stopword,'')
        query = query.replace('  ',' ')
        return query

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []

        # TODO: Extract snippet and the tokens which are not present in the doc.
        query = self.remove_stop_words_from_query(query)
        query_tokens = query.split()
        left_query_tokens = list(set(query.split()))
        doc_tokens = doc.split()
        best_windows = []
        best_windows_start = []

        while len(left_query_tokens)>0:
            token = left_query_tokens.pop()

            if token in doc_tokens:
                indices = [i for i, x in enumerate(doc_tokens) if x == token]
                best_window = None
                best_count = 0
                best_start = 0
                for index in indices:
                    start = max(0, index - self.number_of_words_on_each_side)
                    end = min(len(doc_tokens), index + self.number_of_words_on_each_side + 1)
                    window = doc_tokens[start:end]
                    count = sum([window.count(qt) for qt in query_tokens])
                    if count > best_count:
                        best_window = window
                        best_count = count
                        best_start = start
                best_window = self.bold_query_tokens(best_window, query_tokens, left_query_tokens)

                best_windows.append(best_window)
                best_windows_start.append(best_start)

                
            else:
                not_exist_words.append(token)

        final_snippet = self.create_final_snippet(best_windows,best_windows_start)
        
        
        return final_snippet, not_exist_words
    
    def bold_query_tokens(self, best_window, query_tokens, left_query_tokens):
        ans = []
        
        for token in best_window:
            if token not in query_tokens:
                ans.append(token)
            else:
                if token in left_query_tokens:
                    left_query_tokens.remove(token)
                ans.append('***' + token + '***')
        # print(left_query_tokens)
        return ans
    
    def create_final_snippet(self,best_windows,best_windows_start):
        # print(best_windows_start)
        best_windows = [x for _, x in sorted(zip(best_windows_start, best_windows))]
        best_windows_start.sort()
        ans = ''
        for window in best_windows:
            ans +=  ' '.join(window) + '...'
        ans= ans[:-3]
        return ans
    
if __name__ == '__main__':
    a = Snippet()
    print(a.find_snippet('i am ready ready ready to find it in the best possible way of doing that is for my father mothar ready and daughter in the woods ready','ready mothar'))
