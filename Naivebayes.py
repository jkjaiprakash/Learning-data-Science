# Used for classification
# It is a supervised learning

# It uses Bayes theorem  -----   P(A|B)=(P(B|A).P(A))/P(B) A is an event and B is another event
# When we apply bayes theorem values can get to zero, so we use laplace smoothening to overcome the problem
# To know more refer the photos on 8/1/2023 and mathematical explanation.

# Steps to implementation

# Import Data -----> word count count vectorizer -----> term frequency inverse document frequency------>
# naive bayes classifier -----> Output

from sklearn.feature_extraction.text import CountVectorizer
text=["the bowler throw the ball to batsman","the bowler","the batsman"]
a=CountVectorizer()
a.fit(text)
print(a.vocabulary_)
print(a.get_feature_names())
counts=a.transform(text)
print(counts.toarray())
print(counts.shape)

from sklearn.feature_extraction.text import TfidfTransformer
b=TfidfTransformer()
b.fit(counts)
print(b.idf_)
c=b.transform(counts)
print(c.toarray())

