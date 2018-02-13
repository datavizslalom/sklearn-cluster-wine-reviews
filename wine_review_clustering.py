from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import sklearn
from sklearn import feature_extraction
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
#from sklearn.decomposition import TruncatedSVD
import mpld3
import timeit
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

start = timeit.default_timer()

# Load reviews subset
review_df=pd.read_csv("C:/Users/matt.wheeler/Documents/data/Wine Reviews/winemag-data_95_or_higher.csv", encoding='latin-1')
review_df['review_id'] = review_df.index

print("Wine dataset has this this many Reviews:", len(review_df.index))

# Create Arrays for outputing fields with clustering DF
reviews = review_df.description.tolist()
price = review_df.price.tolist()
points = review_df.points.tolist()
country = review_df.country.tolist()
province = review_df.province.tolist()
region_1 = review_df.region_1.tolist()
region_2 = review_df.region_2.tolist()
variety = review_df.variety.tolist()
winery = review_df.winery.tolist()
review_id = review_df.review_id.tolist()

# stop words to exclude
stopwords = nltk.corpus.stopwords.words('english')

# for breaking words down to roots
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# A tokenizer and stemmer which returns the set of stems in the text that it is passed
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in reviews:
    allwords_stemmed = tokenize_and_stem(i)  # for each item in 'description', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


# tf-df matrix
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, max_features=None,
                                 min_df=0.1, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_only)


tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
print(tfidf_matrix.shape)

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())
tfidf_df.to_csv('tf-idf matrix.csv')


terms = tfidf_vectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

dist_df = pd.DataFrame(dist)

dist_df.to_csv('1-Cosine Simlilarity.csv')


from sklearn.cluster import KMeans

# Test clustering on different number of clusters
range_n_clusters = [3,4,5]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, tfidf_matrix.shape[0] + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10, init='k-means++')
    cluster_labels = clusterer.fit_predict(tfidf_matrix)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(tfidf_matrix, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # Multi-dimensional Scaling
    from sklearn.manifold import MDS
    MDS()

    # two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(pos[:, 0], pos[:, 1], marker='.', s=120, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=100, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()

    print("Top terms per cluster:")
    print()
    order_centroids = clusterer.cluster_centers_.argsort()[:, ::-1]
    for i in range(n_clusters):
         print("Cluster %d Words:" % i, end='')
         for ind in order_centroids[i, :10]:
             print(' %s' % terms[ind], end='')
         print()
         print()


temp = {'text': reviews, 'price': price, 'variety': variety, 'cluster': cluster_labels}
frame = pd.DataFrame(temp, index = [cluster_labels], columns = ['text', 'price', 'variety','cluster'])
print(frame['cluster'].value_counts())

# create data frame that has the result of the MDS plus the cluster numbers and addtional metric
df = pd.DataFrame(dict(review_id=review_id, x=xs, y=ys, price=price, review=reviews, variety=variety, points=points, cluster=cluster_labels, province=province, region_1=region_1,
                       region_2=region_2, country=country, winery=winery))

df.to_csv('Cluster Output (wine reviews).csv', index=False)

stop = timeit.default_timer()
print("Total Seconds Elapsed:", stop - start)