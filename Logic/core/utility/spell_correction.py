class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set(word[i:i+k] for i in range(len(word) - k + 1))

        return shingles
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        return intersection/union

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()
        for doc in all_documents:
            for word in doc.split():
                if word not in all_shingled_words:
                    all_shingled_words[word] = self.shingle_word(word)
                    word_counter[word] = 0
                word_counter[word] += 1

                
        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = list()
        top5_score = list()
        
        limit = 999

        for candidate in self.all_shingled_words:
            score = self.jaccard_score(self.shingle_word(word),self.all_shingled_words[candidate])

            if len(top5_candidates) <=4:
                top5_candidates.append(candidate)    
                top5_score.append(score)
                limit = min(limit, score)
                continue
                
            if score>limit:
                
                index = top5_score.index(limit)
                del top5_score[index]
                del top5_candidates[index]

                top5_candidates.append(candidate)
                top5_score.append(score)

                limit = min(top5_score)

        max_tf = -1
        for candidate in top5_candidates:
            max_tf = max(max_tf, self.word_counter[candidate])
        
        for index, candidate in enumerate(top5_candidates):
            top5_score[index] *= self.word_counter[candidate] / max_tf

        top5_candidates = [x for _, x in sorted(zip(top5_score, top5_candidates))]
        top5_candidates.reverse()
        return top5_candidates
    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = []
        
        words = query.split()
        for word in words:
            final_result.append(self.find_nearest_words(word)[0])

        return ' '.join(final_result)