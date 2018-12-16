import numpy as np
import os
import json
import math
import csv
import collections
from collections import defaultdict
import nltk
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
import spacy

#gloveModel = KeyedVectors.load_word2vec_format("glove.twitter.27B.200d.txt", binary=False,limit=500000)
#glove_model = KeyedVectors.load_word2vec_format("temp.txt", binary=False,limit=500000)

def isUpperCaseWord(word):
    for c in word:
        if (not c.isupper()):
            return False
    return True

def time_to_int(time_in_str):
    time_split = time_in_str.split(':')
    return int(time_split[0])*60 + int(time_split[1])

def getDayBin(df):
    df['day']=df['created'].str.split(' ', 1, expand=True)[1]
    df['day']=df['day'].transform(lambda x: time_to_int(x))
    df['day']=df['day'].transform(lambda x: 0 if (x>500 and x<1140) else 1)
    return df

def transform(text):

    #text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
    vectorizer = CountVectorizer()
# tokenize and build vocab
    vectorizer.fit(text)
# summarize

# encode document
    vector = vectorizer.transform(text)
    return vector


def get_embedding(text, embeddingModel):
    tokens = text.split(' ')
    k = 0
    while True:
        if k >= len(tokens):
            return None
        try:
            if tokens[k]:
                total = embeddingModel.word_vec(tokens[k])
                break
            else:
                k = k + 1
        except KeyError:
            k = k+1
            if k >= len(tokens):
                return None
            continue

    counter = 1
    for i in range(k+1, len(tokens)):
            if tokens[i]:
                try:
                    vec = embeddingModel.word_vec(tokens[i])
                except KeyError:
                    continue
                total = np.add(vec, total)
                counter += 1

    finalVector = total / counter
    print(finalVector)
    return finalVector

def removeSpecialCharacter(word):
    specialCharacters = "!#$."
    for char in specialCharacters:
        word = word.replace(char, "")
    return word

def capitalWordPercentage(wordList, totalWordLength):
    for i in range(len(wordList)):
        wordList[i] = removeSpecialCharacter(wordList[i])
    #print(sum(1 for word in wordList if isUpperCaseWord(word)))
    return sum(1 for word in wordList if isUpperCaseWord(word)) / totalWordLength

def timeOfTweet(time):
    timeRangeList = ["m", "a", "e", "n"]
    # TODO: write logic to extract time range from given time
    print(time)
    return timeRangeList[0]

abbreviatedWordList = [" U ", " B ", " 4 "]

def hasAbbreviatedWords(line):
    for word in abbreviatedWordList:
        if word in line:
            return 1
    return 0

def hashTagPercentage(line, totalWordLength):
    #print(line.count('#'))
    return line.count('#') / totalWordLength

def atTagPercentage(line, totalWordLength):
    #print(line.count('@'))
    return line.count('@') / totalWordLength

def hasHttpLink(line):
    if "http" in line:
        return 1
    return 0

def containsMegyn(line):
    if "megyn" in line or "Megyn" in line:
        return 1
    return 0

def containsCNN(line):
    if "CNN" in line:
        return 1
    return 0

def convertWordToLowerCase(word):
    return word.lower()

def containsThankYou(processedList):
    length = len(processedList)
    for i in range(length):
        if processedList[i] == "thank" and i + 1 < length:
            if processedList[i + 1] == "you":
                return 1
    return 0

def containsMillions(processedList):
    length = len(processedList)
    for i in range(length):
        if processedList[i] == "millions":
            return 1
    return 0

def getSentiment(line):
    # TODO: use nltk to find sentiment - positive / negative
    return "pos"

def getSentenceEmb(sent):
    queE = np.zeros(200)
    count = 0
    for i in range(len(sent)):
        if sent[i] in glove_model.vocab:
            queE += glove_model[sent[i]]
            count += 1
    if count != 0:
        queE = queE / count
    return queE

def getProcessedList(wordList):
    processedList = []
    for i in range(len(wordList)):
        wordList[i] = removeSpecialCharacter(wordList[i])
        wordList[i] = wordList[i].lower()
        processedList.append(wordList[i])
    return processedList


def fillFeatures(df, index):
    # fill features in columns at specified index
    text = df['text'][index]
    wordList = text.split(' ')
    totalWordLength = len(wordList)
    processedList = getProcessedList(wordList)
    df.at[index, 'capitalWordPercentage'] = capitalWordPercentage(wordList, totalWordLength)
    df.at[index, 'hasAbbreviatedWords'] = hasAbbreviatedWords(text)
    df.at[index, 'hashTagPercentage'] = hashTagPercentage(text, totalWordLength)
    df.at[index, 'atTagPercentage'] = atTagPercentage(text, totalWordLength)
    df.at[index, 'hasHttpLink'] = hasHttpLink(text)
    df.at[index, 'containsMegyn'] = containsMegyn(text)
    df.at[index, 'containsCNN']= containsCNN(text)
    df.at[index, 'containsThankYou'] = containsThankYou(processedList)
    df.at[index, 'containsMillions'] = containsMillions(processedList)
    df.at[index, 'totalWords'] = totalWordLength
    #textEmbedding = getSentenceEmb(text)
    #df.at[index, 'textEmbedding'] = np.sqrt(textEmbedding.dot(textEmbedding))
    #df.at[index, 'text'] = getSentenceEmb(text)
    # df['timeOfTweet'][index] = timeOfTweet(df['created'])

    #if df.at[index, 'containsThankYou'] == True:
    #    print(text)
    #if df.at[index, 'containsMillions'] == True:
    #    print(text)


def doML(df):
    X = df.drop('label', axis=1)
    Y = df.label

    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20)
    X_train = X
    y_train = Y

    '''
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  # 'tfidf__use_idf': (True, False),
                  'logisticRegression__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                  }

    # pipe = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression())
    pipe = make_pipeline(CountVectorizer(stop_words='english'), LogisticRegression())
    param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10,
                                            100]}  # , "tfidfvectorizer__ngram_range": [(1,1), (1,2), (1,3)]}
    gs_clf_logReg = GridSearchCV(pipe, param_grid, cv=5)

    gs_clf_logReg = gs_clf_logReg.fit(X_train.text, y_train)

    print('logReg:', gs_clf_logReg.best_score_)
    print('logReg:', gs_clf_logReg.best_params_)
    '''

    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3)}

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.10)

    text_clf_logReg = Pipeline([('vect', CountVectorizer(stop_words='english')),
                                 ('tfidf', TfidfTransformer()),
                                ('clf', LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'))])

    text_clf_logReg = text_clf_logReg.fit(X_train, Y_train)

    preds_logReg = text_clf_logReg.predict(X_valid)
    print(preds_logReg)
    accuracy_logReg = np.mean(preds_logReg == Y_valid)
    print('Accuracy_logReg:', accuracy_logReg)

    model = gs_clf_logReg

    X_test = pd.read_csv('../data/test.csv')
    preds = model.predict(X_test.text)
    print(preds)
    preds_df = pd.DataFrame(preds)
    preds_df.to_csv('logistic_regression.csv')


def testML(df):
    # TODO
    # Reference: https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/
    X = df.drop('label', axis=1)
    y = df.label

    # 2. instantiate model
    logreg = LogisticRegression()

    # 3. fit
    logreg.fit(X, y)


def convertDFToVector(df, featureColumnList, countVectors):
    numberOfDataPoints, numberOfFeatures = df.shape
    vecRow, vecCol = countVectors.shape
    print(numberOfDataPoints, numberOfFeatures)
    print(vecRow, vecCol)
    featureVector = np.zeros((numberOfDataPoints, numberOfFeatures + vecCol))

    for index, row in df.iterrows():
        featureList = []
        f = 0
        #sentenceEmbedding = df[index]['text']
        for featureColumn in featureColumnList:
            #featureList.append(df.at[index, featureColumn])
            featureVector[index][f] = df.at[index, featureColumn]
            f += 1
        for col in range(vecCol):
            featureVector[index][f] = countVectors.at[index, col]
        #featureVector = np.vstack([featureVector, featureList])
    #print(featureVector)
    return featureVector

def doML(trainDf, testDf, trainCountVectors, testCountVectors):
    # Removed textEmbedding
    featureColumns = ['favoriteCount', 'retweetCount', 'capitalWordPercentage',
                      'hashTagPercentage', 'atTagPercentage', 'hasHttpLink',
                      'containsMegyn', 'containsCNN', 'containsThankYou',
                      'totalWords', 'day']

    #countVectors.columns = vectorizer.get_feature_names()

    #X = trainDf.loc[:, featureColumns]
    #print(X.shape)

    X = convertDFToVector(trainDf.loc[:, featureColumns], featureColumns, trainCountVectors)

    Y = df.label

    #
    # evaluate on development set
    #
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.10)
    #model = LogisticRegression()
    model = RandomForestClassifier(n_estimators=600, criterion='entropy', max_depth=100, min_samples_split=2, max_features='sqrt')
    #model = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=60)
    #model.fit(X, Y)

    model.fit(X_train, Y_train)
    y_predicted = model.predict(X_valid)
    accuracyOfTraining = np.mean(y_predicted == Y_valid)
    print('Accuracy of Training:', accuracyOfTraining)


    '''
    grid_search_params = {'bootstrap': [True],
     'max_depth': [40, 50, 60, 70, 80, 90, 100],
     'max_features': ['auto', 'sqrt'],
     #'min_samples_leaf': [1, 2, 4],
     #'min_samples_split': [2, 5, 10],
     'n_estimators': [200, 400, 600, 800, 1000, 1200]}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    #rf_random = RandomizedSearchCV(estimator=rf, param_distributions=grid_search_params, n_iter=100, cv=3, verbose=2,
    #                              random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(estimator=rf, param_grid=grid_search_params,
                               cv=3, n_jobs=-1, verbose=2)
    # Fit the random search model
    grid_search.fit(X, Y)

    print(grid_search.best_params_)
    #print(rf_random.best_params_)
    best_grid = grid_search.best_estimator_
    print(best_grid)
    
    y_predicted = rf_random.predict(X_valid)
    '''

    # Create a SVC classifier using an RBF kernel
    #svm = SVC(kernel='rbf', random_state=0, gamma=.01, C=10)

    # Train the classifier
    #svm.fit(X, Y)

    #y_predicted = model.predict(X_valid)
    #y_predicted = rf_random.predict(X_valid)
    #accuracyOfTraining = np.mean(y_predicted == Y_valid)
    #print('Accuracy of Training:', accuracyOfTraining)

    #X_test = testDf.loc[:, featureColumns]
    X_test = convertDFToVector(testDf.loc[:, featureColumns], featureColumns, testCountVectors)

    #
    # evaluate on test set
    #
    #model.fit(X, Y)

    #print("Important model features")
    #print(model.feature_importances_)
    #featureImportances = model.feature_importances_
    #print(featureImportances)
    #print(dict(zip(featureColumns, featureImportances)))
    #new_pred_class = model.predict(X_test)
    #new_pred_class = grid_search.predict(X_test)
    new_pred_class = model.predict(X_test)
    #print(new_pred_class)


    preds_df = pd.DataFrame(new_pred_class)
    preds_df.to_csv('random_forest_grid_search.csv')

    #kaggle_data = pd.DataFrame({'id': test.PassengerId, 'label': new_pred_class}).set_index('id')
    #kaggle_data.to_csv('temp.csv')



def addNewFeatures(df, isTestDf):

    '''
    Major Features so far:
    1. capitalWordPercentage
    2. timeOfTweet (use created column and convert into morning, afternoon, evening, night)
    3. hasAbbreviatedWords
    4. favoriteCount
    5. retweetCount
    6. # tags percentage in tweet
    7. @ tags percentage in tweet
    8. hasHttpLink
    9. containsMegyn
    10. containsCNN
    11. containsThankYou
    12. containsMillions
    13. totalWords

    TODO: Do a test of feature co-relation with prediction
    TODO: Identify highest co-related words with the prediction
    TODO: There is only 1 megyn in test set. So, should I include it?
    TODO: There is only 1 abbreviated word in test set. So, should I include it?

    TODO:

    1. russian
    2.
    3. weekday/weekend

    Decide whether to include following features (value exists rarely):
    1. id
    2. latitude & longitude
    3. replyToUID
    4. replyToSID

    Decide whether to add following features:
    1. hasSpecialCharacters (!?)
    2. tweetSentiment (positive, negative)
    3. word/character count in tweet greater than some threshold
    '''
    # TODO: Decide whether to include features where values are included only once or twice
    if isTestDf:
        df.drop(columns=['screenName', 'isRetweet', 'retweeted'])
    else:
        df.drop(columns = ['statusSource', 'screenName', 'isRetweet', 'retweeted'])

    # Add new features
    df['capitalWordPercentage'] = 0.0
    df['hasAbbreviatedWords'] = 0
    df['hashTagPercentage'] = 0.0
    df['atTagPercentage'] = 0.0
    df['hasHttpLink'] = 0
    df['containsMegyn'] = 0
    df['containsCNN'] = 0
    df['containsThankYou'] = 0
    df['containsMillions'] = 0
    df['totalWords'] = 0
    df['textEmbedding'] = 0
    # TODO: Decide whether to classify the timings
    # df['timeOfTweet'] = None

if __name__ == "__main__":
    #print(getSentenceEmb("This is a test for sentence embedding"))
    #print(getSentenceEmb("This is a test for randome sentence"))

    # NOTE: Change the location of file
    #df = pd.read_csv('../data/train.csv')
    df = pd.read_csv('train.csv')

    # add and remove features in dataframe
    addNewFeatures(df, False)

    # add value for each feature by processing the text
    for index, row in df.iterrows():
        fillFeatures(df, index)

    df = getDayBin(df)

    #vectorizer = CountVectorizer(stop_words='english')
    tfidfVectorizer = TfidfVectorizer()

    #vectorizer = CountVectorizer(stop_words='english')
    #tfidfVectorizer = TfidfTransformer()
    trainCountVectors = pd.DataFrame(tfidfVectorizer.fit_transform(df.text.tolist()).toarray())

    #corpus = df.text.tolist()
    #vectorizer = CountVectorizer()
    #vectorModel = vectorizer.fit(corpus)

    #print(isUpperCaseWord("MAdE"))
    #print(capitalWordPercentage(["MAKE", "India", "holy", "AgAIN"], 4))
    #print(removeSpecialCharacter("AGAIN."))
    #print(hasHttpLink("Thank you Indiana! #MakeAmericaGreatAgain https://t.co/pxvSL8cs3B"))
    #print(hasAbbreviatedWords("Do it too 4 too"))
    #print(df['capitalWordPercentage'])

    # apply ML algorithm with the features in dataframe
    #doML(df)

    # NOTE: Change the location of file
    #test_df = pd.read_csv('../data/test.csv')
    test_df = pd.read_csv('test.csv')

    # add and remove features in dataframe
    addNewFeatures(test_df, True)

    # add value for each feature by processing the text
    for index, row in test_df.iterrows():
        fillFeatures(test_df, index)

    test_df = getDayBin(test_df)

    #vectorizer = CountVectorizer(stop_words='english')
    #tfidfVectorizer = TfidfVectorizer(vectorizer)
    testCountVectors = pd.DataFrame(tfidfVectorizer.transform(test_df.text.tolist()).toarray())

    #print(test_df['capitalWordPercentage'])
    doML(df, test_df, trainCountVectors, testCountVectors)
    #testML(df)





