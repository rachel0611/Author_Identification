import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics, model_selection, naive_bayes

class MVNB:
  def __init__(self):
    self.prepare_data()

  def prepare_data(self):
    train = pd.read_csv('train.csv', encoding="latin1")
    test = pd.read_csv('test.csv', encoding="latin1")
    pd.set_option('max_colwidth', 500)
    train.text= train.text.astype(str)
    train.author = pd.Categorical(train.author)
    self.train = train
    author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2, 'SK':3, 'BS':4}
    self.train_y = train['author'].map(author_mapping_dict)
    train_id = train['id'].values
    test_id = test['id'].values
    cols_to_drop = ['id', 'text']
    self.train_X = train.drop(cols_to_drop+['author'], axis=1)
    self.test_X = test.drop(cols_to_drop, axis=1)
    self.count_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))
    full_count = self.count_vec.fit(train['text'].values.astype('U'))
    self.train_count = self.count_vec.transform(train['text'].values.astype('U'))
    self.test_count = self.count_vec.transform(test['text'].values.astype('U'))

  def runMNB(self, train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

  def trainModel(self):
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([self.train.shape[0], 5])
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(self.train_X):
      dev_X, val_X = self.train_count[dev_index], self.train_count[val_index]
      dev_y, val_y = self.train_y[dev_index], self.train_y[val_index]
      pred_val_y, pred_test_y, model = self.runMNB(dev_X, dev_y, val_X, val_y, self.test_count)
      pred_full_test = pred_full_test + pred_test_y
      pred_train[val_index,:] = pred_val_y
      cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Mean CV Score:", np.mean(cv_scores))
    self.model = model
  
  def predict(self, text):
    input_text = [text]
    data_in = pd.DataFrame()
    data_in['text'] = input_text
    input_count = self.count_vec.transform(data_in['text'])
    res = self.model.predict_proba(input_count)
    max_prob = np.max(res)
    author_index = np.argmax(res,axis=1)[0]
    author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2, 'SK':3, 'BS':4}
    if max_prob > 0.8:
      return "Author is " + str(list(author_mapping_dict.keys())[list(author_mapping_dict.values()).index(author_index)])
    else:
      return "No Author Found!"

