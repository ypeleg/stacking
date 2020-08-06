
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

def label_var(data,variables_cat):
    lb=[]
    for m in variables_cat:
        l=LabelEncoder()
        lb.append(l.fit(list(data[m].dropna())))
    return lb

def label_enc(data,l,categorical_features):
    i=0
    for m in categorical_features:
        data.loc[data[m].notnull(),m]=l[i].transform(data.loc[data[m].notnull(),m])
        i=i+1

df_tr = pd.read_csv("training_v2.csv")
df_ts = pd.read_csv("unlabeled.csv")

def lazy_conc(train, test):
    return pd.concat(objs=[train, test], axis=0)

def lazy_split(df, train, test):
    test = copy.copy(df[len(train):])
    train = copy.copy(df[:len(train)])
    return train, test

df = lazy_conc(df_tr, df_ts)
df['hospital_admit_source'] = df['hospital_admit_source'].replace({'Other ICU': 'ICU','ICU to SDU':'SDU', 'Step-Down Unit (SDU)': 'SDU', 'Other Hospital':'Other','Observation': 'Recovery Room','Acute Care/Floor': 'Acute Care'})
df['apache_2_bodysystem'] = df['apache_2_bodysystem'].replace({'Undefined diagnoses': 'Undefined Diagnoses'})
df = lazy_split(df, df_tr, df_ts)




train_columns = [x for x in df_tr.columns if x not in ['encounter_id','patient_id','hospital_death','readmission_status']]
categorical_features = [c for c in df_tr.columns if df_tr[c].dtypes=='object']

df_tr_ts = pd.concat([df_tr[categorical_features],df_ts[categorical_features]])

for m in categorical_features:
    lb = LabelEncoder().fit(list(df_tr_ts[m].dropna()))
    df_tr.loc[df_tr[m].notnull(),m]=lb.transform(df_tr.loc[df_tr[m].notnull(),m])
    df_ts.loc[df_ts[m].notnull(), m] = lb.transform(df_ts.loc[df_ts[m].notnull(), m])
    df_tr[m] = df_tr[m].astype(float)
    df_ts[m] = df_ts[m].astype(float)

categorical_index = [train_columns.index(x) for x in categorical_features]
target = df_tr['hospital_death']

for df in [df_tr, df_ts]:
    for m in categorical_features:
        df[m] = df[m].astype('str')
y = df_tr['hospital_death']
enc_id = df_ts['encounter_id']
to_drop = ['encounter_id','patient_id','hospital_death','readmission_status']
train = df_tr.copy()
test = df_ts.copy()
categoricals_features = [i for i in list(set(train.columns) - set(train._get_numeric_data().columns)) if i not in to_drop]
non_categorical_features = [i for i in list(set(train._get_numeric_data().columns)) if i not in to_drop]
features = [col for col in train.columns if col not in to_drop ]
prev_categs = categoricals_features
df_tr = train.copy()
df_ts = test.copy()












from sklearn import metrics
from ypeleg.sababa_models import SababaLGBClassifier

FOLDS = 5

y_train = df_tr['hospital_death']
df_tr = df_tr[features]
df_ts = df_ts[features]
df_tr['hospital_death'] = y_train
df_tr.to_csv('train.csv', index=False)
df_ts.to_csv('test.csv', index=False)


from sklearn import metrics
from ypeleg.sababa_models import SababaLGBClassifier

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
    cur_auc = metrics.roc_auc_score(y_val, oof_pred[val_ind])
    print('Current AUC: ', cur_auc)

score = roc_auc_score(y_train, oof_pred)
print('OOF AUC score: ', score)


sub = pd.read_csv('sample_submission.csv')
sub['hospital_death'] = preds
sub.to_csv('submission.csv')

