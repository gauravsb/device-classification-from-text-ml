from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np



df = pd.read_csv('train.csv')
X=df.drop('label', axis=1)
y=df.label

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20)
#X_train=X
#y_train=y

'''
#These parameters are used for NB and SVM
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3)}
'''

#Multinomial Naive Bayes

'''
text_clf_nb = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),])


gs_clf_nb = GridSearchCV(text_clf_nb, parameters, n_jobs=-1)
gs_clf_nb = gs_clf_nb.fit(X_train.text, y_train)

print('nb:', gs_clf_nb.best_score_)
print('nb:', gs_clf_nb.best_params_)
preds_nb = gs_clf_nb.predict(X_valid.text)
accuracy_nb = np.mean(preds_nb == y_valid)
print('Accuracy_nb 1:', accuracy_nb)
'''

'''
text_clf_nb = text_clf_nb.fit(X_train.text, y_train)
preds_nb = text_clf_nb.predict(X_valid.text)
accuracy_nb = np.mean(preds_nb == y_valid)
print('Accuracy_nb 2:', accuracy_nb)
'''


#SVM
'''
text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42)),])

gs_clf_svm = GridSearchCV(text_clf_svm, parameters, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(X_train.text, y_train)

print('svm:', gs_clf_svm.best_score_)
print('svm:', gs_clf_svm.best_params_)

#text_clf_svm = text_clf_svm.fit(X_train.text, y_train)
preds_svm = gs_clf_svm.predict(X_valid.text)
accuracy_svm = np.mean(preds_svm == y_valid)
print('Accuracy_svm:', accuracy_svm)

'''

#Logistic regression
'''
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              #'tfidf__use_idf': (True, False),
              'logisticRegression__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
              }

text_clf_logReg = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         #('tfidf', TfidfTransformer()),
                         ('clf', LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'))])

gs_clf_logReg = GridSearchCV(text_clf_logReg, parameters, n_jobs=-1)

'''

'''
#pipe = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression())
pipe = make_pipeline(CountVectorizer(stop_words='english'), LogisticRegression())
param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100]}#, "tfidfvectorizer__ngram_range": [(1,1), (1,2), (1,3)]}
gs_clf_logReg = GridSearchCV(pipe, param_grid, cv=5)


gs_clf_logReg = gs_clf_logReg.fit(X_train.text, y_train)

print('logReg:', gs_clf_logReg.best_score_)
print('logReg:', gs_clf_logReg.best_params_)

#text_clf_logReg = text_clf_logReg.fit(X_train.text, y_train)

#preds_logReg = gs_clf_logReg.predict(X_valid.text)
#accuracy_logReg = np.mean(preds_logReg == y_valid)
#print('Accuracy_logReg:', accuracy_logReg)
'''

#Random Forest

param_grid = { 
    'clf__n_estimators': [1, 1000],
    'clf__max_features': ['auto', 'sqrt', 'log2'],
    #'clf__max_depth' : [4,5,6,7,8],
    #'clf__criterion' :["gini", "entropy"]
}

text_clf_rf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf', RandomForestRegressor())])

gs_clf_rf = GridSearchCV(text_clf_rf, param_grid, cv=5)

#Here is where the features of the training data are being chosen. Since the pipeline uses tf idf, can't say what the result of passing in non textual data will mean
gs_clf_rf.fit(X_train.text, y_train)
preds_rf = np.sign(gs_clf_rf.predict(X_valid.text))
accuracy_rf = np.mean(preds_rf == y_valid)
print('Accuracy RF:', accuracy_rf)


#Improvements can be made by doing the above with stemming

model=gs_clf_rf

X_test = pd.read_csv('test.csv')
preds = model.predict(X_test.text)
print(preds)
preds_df=np.sign(pd.DataFrame(preds))
preds_df.to_csv('ans_randomForest_5.csv')

