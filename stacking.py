


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from ypeleg.sababa_models import SababaLGBClassifier

FOLDS = 5

df_tr = pd.read_csv('train.csv')
df_ts = pd.read_csv('test.csv')
y_train = df_tr['hospital_death']
df_tr.drop('hospital_death', axis=1, inplace=True)

preds = np.zeros((df_ts.shape[0], 1))
oof_pred = np.zeros((df_tr.shape[0], 1))

for tr_ind, val_ind in StratifiedKFold(FOLDS).split(df_tr, y_train):
    x_train, x_val = df_tr.iloc[tr_ind], df_tr.iloc[val_ind]
    y_tr, y_val = y_train.values[tr_ind], y_train.values[val_ind]
    model = SababaLGBClassifier()
    model.fit(x_train, y_tr, x_val, y_val)
    preds[:, 0] += model.predict(df_ts) / FOLDS
    oof_pred[val_ind] = np.expand_dims(model.predict(x_val), 1)
    print('Current AUC: ', metrics.roc_auc_score(y_val, oof_pred[val_ind]))

print('OOF AUC score: ', roc_auc_score(y_train, oof_pred))

sub = pd.read_csv('sample_submission.csv')
sub['hospital_death'] = preds
sub.to_csv('submission.csv')




