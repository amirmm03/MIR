from .graph import LinkGraph
# from ..indexer.indexes_enum import Indexes
# from ..indexer.index_reader import Index_reader

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []
        self.authorities = []
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            self.graph.add_node(movie['id'])
            # self.hubs.append({})
            for star in movie['stars']:
                if not self.graph.has_node(star):
                    self.graph.add_node(star)
                self.graph.add_edge(movie['id'], star)

        # self.hubs = {node: 1 for node in self.graph.nodes}
        # self.authorities = {node: 1 for node in self.graph.nodes}


    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        root_star = set()
        for movie in self.root_set:
            for star in movie:
                root_star.add(star)

        for movie in corpus:
            for star in movie['stars']:
                if star in root_set:
                    if not self.graph.has_node(movie['id']):
                        self.graph.add_node(movie['id'])
                    self.graph.add_edge(movie['id'], star)
                    
        # print(self.graph.graph.edges)

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        
        import networkx as nx
        h_s, a_s = nx.hits(self.graph.graph, max_iter=num_iteration)
        
        sorted_hubs = sorted(h_s, key=h_s.get, reverse=True)[:max_result]
        sorted_authorities = sorted(a_s, key=a_s.get, reverse=True)[:max_result]
        print(sorted(a_s, key=a_s.get, reverse=True))


        return sorted_authorities, sorted_hubs

if __name__ == "__main__":
    import json
    # You can use this section to run and test the results of your link analyzer
    json_file_path = './IMDB_preprocessed.json'
    with open(json_file_path, "r") as file:
        corpus = json.load(file)
    # corpus = []  it shoud be your crawled data
    root_set = corpus[0:100]   # it shoud be a subset of your corpus

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
