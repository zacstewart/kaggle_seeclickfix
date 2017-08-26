import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import KFold

PREDICTABLES = ['num_votes', 'num_views', 'num_comments']

CV = True

SCALE = {
    'num_votes':     np.log1p,
    'num_comments':  np.log1p,
    'num_views':     np.log1p
}

UNSCALE = {
    'num_votes':     np.expm1,
    'num_comments':  np.expm1,
    'num_views':     np.expm1
}

train = pd.io.parsers.read_csv('data/train.csv', parse_dates = ['created_time'], encoding = 'utf-8')
test  = pd.io.parsers.read_csv('data/test.csv', parse_dates = ['created_time'],encoding = 'utf-8')

# Drop first 10 months
train = train[train['created_time'] > pd.to_datetime('2012-11-1')]

class FactorExtractor:
  def __init__(self, factor):
    self.factor = factor

  def transform(self, data):
    return [{self.factor: self.normalize(tt)} for tt in data[self.factor]]

  def fit(self, *_):
    return self

  def normalize(self, tag):
    if type(tag) != str: tag = '_MISSING_'
    return tag

class LocationExtractor:
  def transform(self, data):
    return np.asarray(data[['latitude', 'longitude']])

  def fit(self, *_):
    return self

class TextExtractor:
  def __init__(self, column):
    self.column = column

  def transform(self, data):
    return np.asarray(data[self.column]).astype('U')

  def fit(self, *_):
    return self

class LengthVectorizer:
  VEC_LEN = np.vectorize(len)

  def transform(self, data):
    return self.VEC_LEN(data).astype(float)

  def fit(self, *_):
    return self;

class ArrayUpDimension:
  def transform(self, data):
    return data.reshape((data.shape[0], 1))

  def fit(self, *_):
    return self

class DateExtractor:
  def transform(self, data):
    return data['created_time']

  def fit(self, *_):
    return self

class MonthsSinceCreated:
  def __init__(self, end_date):
    self.end_date = end_date

  def transform(self, data):
    return np.array(map(self.difference, data))

  def difference(self, date):
    months = self.end_date.month - date.month
    years  = self.end_date.year - date.year
    return 12 * years + months

  def fit(self, *_):
    return self

class MiniBatchKMeansTransformer:
  def __init__(self, n_clusters):
    self.model = MiniBatchKMeans(n_clusters = n_clusters)

  def fit(self, data, _):
    self.model.fit(data)
    return self

  def transform(self, data):
    return self.model.predict(data)

tag_featurizer = Pipeline([
  ('tag_type_extractor',  FactorExtractor('tag_type')),
  ('dict_vectorizer',     DictVectorizer(sparse = False))
])

source_featurizer = Pipeline([
  ('source_extractor',  FactorExtractor('source')),
  ('dict_vectorizer',   DictVectorizer(sparse = False))
])

location_featurizer = Pipeline([
  ('location_extractor',  LocationExtractor()),
  ('kmeans',              MiniBatchKMeansTransformer(n_clusters = 4)),
  ('updim',               ArrayUpDimension()),
  ('onehot',              OneHotEncoder())
])

desc_length_featurizer = Pipeline([
  ('desc_extractor',  TextExtractor('description')),
  ('len_vectorizer',  LengthVectorizer()),
  ('scaler',          StandardScaler()),
  ('updim_array',     ArrayUpDimension())
])

desc_ngrams_featurizer = Pipeline([
  ('desc_extractor',    TextExtractor('description')),
  ('count_vectorizer',  CountVectorizer(ngram_range = (1, 3), stop_words = 'english', encoding = 'cp1252')),
  ('tfidf_transformer', TfidfTransformer())
])

summary_ngrams_featurizer = Pipeline([
  ('summary_extractor', TextExtractor('summary')),
  ('count_vectorizer',  CountVectorizer(ngram_range = (1, 3), stop_words = 'english', encoding = 'cp1252')),
  ('tfidf_transformer', TfidfTransformer())
])

months_since_created = Pipeline([
  ('date_extractor',  DateExtractor()),
  ('months_since',    MonthsSinceCreated(pd.to_datetime('2013-09-20'))),
])

features = FeatureUnion([
  ('tag_features',            tag_featurizer),
  ('source_featurs',          source_featurizer),
  ('location_featurizer',     location_featurizer),
  ('desc_length_featurizer',  desc_length_featurizer),
  ('desc_tfidf_ngrams',       desc_ngrams_featurizer),
  ('summary_tfidft_ngrams',   summary_ngrams_featurizer)
])

predictor = SGDRegressor(shuffle = True, verbose = 1)

pipeline = Pipeline([
  ('feature_union',  features),
  ('predictor',      predictor)
])

import pdb; pdb.set_trace()

if CV:
  k_fold = KFold(train.shape[0], 10)

  scores = dict(zip(PREDICTABLES, [[], [], []]))
  fold_n = 0

  for construct_idx, validate_idx in k_fold:
    fold_n += 1
    print 'Fold %d' % fold_n
    construct = train.iloc[construct_idx]
    validate  = train.iloc[validate_idx]

    for predictable in PREDICTABLES:
      construct_months = months_since_created.transform(construct)
      construct_targets = construct[predictable]
      construct_targets = SCALE[predictable](construct_targets)

      validate_months = months_since_created.transform(validate)
      validate_targets = validate[predictable]
      validate_targets = SCALE[predictable](validate_targets)

      pipeline.fit(construct, construct_targets)
      score = pipeline.score(validate, validate_targets)
      scores[predictable].append(score)

  for predictable in PREDICTABLES:
    print '%s: %f' % (predictable, (sum(scores[predictable]) / len(scores[predictable])))

submission = pd.DataFrame({'id': test['id']})

print "Building submission"
for predictable in PREDICTABLES:
  train_target = train[predictable]
  train_target = SCALE[predictable](train_target)

  pipeline.fit(train, train_target)
  predictions = pipeline.predict(test)
  predictions = UNSCALE[predictable](predictions)
  predictions[predictions < 0] = 0.0
  submission[predictable] = predictions

submission.to_csv('submission.txt', index = False)

import pdb; pdb.set_trace()
