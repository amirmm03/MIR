import numpy as np
import itertools
import random


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set()

        words = document.split()
        for i in range(len(words) - k + 1):
            shingle = ' '.join(words[i:i+k])
            shingles.add(shingle)

        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        
        k = 2
        all_shingles = set()
        for document in self.documents:
            shingles = self.shingle_document(document, k)
            all_shingles.update(shingles)
        
        char_matrix = np.zeros((len(all_shingles), len(self.documents)), dtype=int)
        all_shingles = list(all_shingles)
        for j, document in enumerate(self.documents):
            shingles = self.shingle_document(document, k)
            for shingle in shingles:
                i = all_shingles.index(shingle)
                char_matrix[i][j] = 1


        return char_matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        
        char_matrix = self.build_characteristic_matrix()
        # print('befpar')
        permutaions = [np.random.permutation(len(char_matrix)) for _ in range(self.num_hashes)]
        # print('aftpar')



        signatures = np.full((self.num_hashes, len(self.documents)), np.inf)

        for i in range(len(char_matrix)):
            for j in range(len(char_matrix[0])):
                if char_matrix[i, j] == 1:
                    for k in range(self.num_hashes):
                        hash_value = permutaions[k][i]
                        if hash_value < signatures[k, j]:
                            signatures[k, j] = hash_value

        return signatures


    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """

        ans = {}
        for b in range(bands):
            for j in range(len(self.documents)):
                
                band = signature[b*rows_per_band:(b+1)*rows_per_band, j]
                hash_value = hash(tuple(band))

                
                if hash_value not in ans:
                    ans[hash_value] = []
                ans[hash_value].append(j)
        return ans

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        num_bands = 25
        signature = self.min_hash_signature()
        
        ans = self.lsh_buckets(signature, num_bands, self.num_hashes//num_bands)
        return ans

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        return intersection/union

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)



def main():
    a = MinHashLSH(None,None)
    ans = a.shingle_document('i am not testing',2)
    print(ans)


    import json
    json_file_path = '.\Logic\core\LSHFakeData.json'
    with open(json_file_path, "r") as file:
        data = json.load(file)
    print('len is ', len(data))
    docs = []
    for i in data:
        docs.append(' '.join(i['summaries']))

    a = MinHashLSH(docs,100)
    buckets = a.perform_lsh()
    a.jaccard_similarity_test(buckets,docs)

    ans = set()
    for i in buckets:
        if len(buckets[i])>1:
            ans.add(tuple(buckets[i]))
    print(ans,len(ans))
    
    json_file_path = "./IMDB_crawled.json"
    with open(json_file_path, "r") as file:
        data += json.load(file)
    print('len is ', len(data))
    docs = []
    for i in data:
        docs.append(' '.join(i['summaries']))
    a = MinHashLSH(docs[0:20],100)
    buckets = a.perform_lsh()
    print('created')
    a.jaccard_similarity_test(buckets,docs[0:20])
    


if __name__ == '__main__':
    main()
