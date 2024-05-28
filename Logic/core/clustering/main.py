import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# import wandb

import sys
sys.path.append(os.path.abspath(os.path.join('Logic', 'core')))

from word_embedding.fasttext_data_loader import FastTextDataLoader
from word_embedding.fasttext_model import FastText
from .dimension_reduction import DimensionReduction
from .clustering_metrics import ClusteringMetrics
from .clustering_utils import ClusteringUtils
if __name__ == '__main__':
    # Main Function: Clustering Tasks

    # 0. Embedding Extraction
    # TODO: Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.

    project_name = 'MIR IMDB'
    def nothing(a):
        return a

    ft_model = FastText(method='skipgram')
    ft_model.prepare(None, mode="load")

    ft_data_loader = FastTextDataLoader(preprocessor=nothing,file_path='IMDB_crawled.json')

    X, y = ft_data_loader.create_train_data()
    X = X[0:200]
    y = y[0:200]

    X_emb = np.array([ft_model.get_doc_embedding(' '.join(text)) for text in tqdm(X)])

    # 1. Dimension Reduction
    # TODO: Perform Principal Component Analysis (PCA):
    #     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
    #     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
    #     - Draw plots to visualize the results.
    dr = DimensionReduction()
    X = dr.pca_reduce_dimension(X_emb, 50)
    dr.wandb_plot_explained_variance_by_components(X_emb, project_name, "1")

    # clustering = ClusteringUtils()
    # clustering.visualize_elbow_method_wcss(X, [2 * i for i in range(1, 10)], project_name, "6")
    # input('aaaaa')


    # TODO: Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
    #     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
    #     - Use the output vectors from this step to draw the diagram.

    #  delete this coment!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    x_2d = dr.convert_to_2d_tsne(X)
    dr.wandb_plot_2d_tsne(X, project_name, "2")



    # 2. Clustering
    ## K-Means Clustering
    # TODO: Implement the K-means clustering algorithm from scratch.
    # TODO: Create document clusters using K-Means.
    # TODO: Run the algorithm with several different values of k.
    # TODO: For each run:
    #     - Determine the genre of each cluster based on the number of documents in each cluster.
    #     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
    #     - Check the implementation and efficiency of the algorithm in clustering similar documents.
    # TODO: Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
    # TODO: Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)


    centeroids = []
    cluster_assignments = []
    clustering = ClusteringUtils()
    ks = [ i*2 for i in range(1,3)]
    for k in ks:
        print('k is ', k)
        clustering.visualize_kmeans_clustering_wandb(X, k, project_name, "3")
    print('bef')
    clustering.plot_kmeans_cluster_scores(X, y, ks, project_name, "4")
    print('after')
    ## Hierarchical Clustering
    # TODO: Perform hierarchical clustering with all different linkage methods.
    # TODO: Visualize the results.

    for method in ["average",  "single", "ward", "complete"]:
        clustering.wandb_plot_hierarchical_clustering_dendrogram(X, project_name, method, "5")

    # 3. Evaluation
    # TODO: Using clustering metrics, evaluate how well your clustering method is performing.
    clustering.visualize_elbow_method_wcss(X, [2 * i for i in range(1, 10)], project_name, "6")
    metric = ClusteringMetrics()
    for k in range(2, 20, 3):
        centeroids,cluster_assignments = clustering.cluster_kmeans(X, k)

        print(f"for {k}: ARS: {metric.adjusted_rand_score(y, cluster_assignments)}")
        print('---------------------------')
        print(f"pure: {metric.purity_score(y, cluster_assignments)}")
        print('---------------------------')
        print(f"silhouette: {metric.silhouette_score(X, cluster_assignments)}")
        print('---------------------------')
