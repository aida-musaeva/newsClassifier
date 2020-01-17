import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

stopwords = nltk.corpus.stopwords.words('russian')

news = []
category = []
test = []

with open('news_train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        row = line.strip().split('\t', maxsplit=2)
        category.append(row[0])
        news.append(row[2])

with open('news_test.txt', 'r', encoding='utf-8') as f:
    for line in f:
        row = line.strip().split('\t', maxsplit=1)
        test.append(row[1])

i = 0
tfidf = TfidfVectorizer(stop_words=stopwords, max_features=80000, smooth_idf=True)
train = tfidf.fit_transform(news)

cls = LogisticRegression(C=7.3, class_weight='balanced', dual=False,
        fit_intercept=True, intercept_scaling=1, max_iter=300,
        n_jobs=1, penalty='l2',
        random_state=8, tol=0.0001, verbose=0,
        warm_start=False)


cls.fit(train, category)

out_file = open('output.txt', 'w')

for data in test:
    vec = tfidf.transform([data]).toarray()
    pred_category = str(cls.predict(vec))
    out_file.writelines("%s\n" % pred_category[2:len(pred_category) - 2])
out_file.close()
