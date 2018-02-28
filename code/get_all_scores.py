from __future__ import division
import preprocessing as pre
from sklearn.metrics import roc_auc_score
import glob
import joblib

train_text, train_y = pre.load_data()

models = glob.glob('../predictions/cval*')

model_dict = {model.split('/')[-1].split('.')[0] : roc_auc_score(train_y, joblib.load(model)) for model in models}
