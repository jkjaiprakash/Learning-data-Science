import sklearn.datasets as skd
categories=['alt.athesim','comp.graphics','sci.med','soc.relegion.christian']
train=skd.load_files(r"C:\Course_Python\DataScience\New folder\20news-bydate\20news-bydate-train",categories=categories,encoding="ISO-8859-1")
test=skd.load_files(r"C:\Course_Python\DataScience\New folder\20news-bydate\20news-bydate-test",categories=categories,encoding="ISO-8859-1")
print(train.keys())
print(train["target"])

from sklearn.feature_extraction.text import CountVectorizer
a=CountVectorizer()
b=a.fit_transform(train.data)
print(b.shape)

from sklearn.feature_extraction.text import TfidfTransformer
c=TfidfTransformer()
d=c.fit_transform(b)
print(b.shape)

from sklearn.naive_bayes import MultinomialNB
e=MultinomialNB().fit(d,train.target)
x=a.transform(test.data)
y=c.transform(x)
predicted=e.predict(y)
print(predicted)

from sklearn.metrics import accuracy_score, classification_report
accuracy= accuracy_score(test.target,predicted)
print(accuracy)

answer=classification_report(test.target,predicted)
print(answer)
