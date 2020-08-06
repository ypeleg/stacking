
lgb_params = {
    'learning_rate': 0.01,
    'max_depth': 5,
    'feature_fraction': 0.1,
    'bagging_fraction': 0.75,
    'use_missing': True,
    'bagging_freq': 1,
    'num_leaves': 15,
    'task': 'train',
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'n_jobs': -1,
}

class LGBClassifier(object):

    def __init__(self):
        pass

    def sababa_params(self):
        params = lgb_params
        params.update(
            {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            })
        return params

    def fit(self, X, y, x_val=None, y_val=None, n_itter=5000):
        import lightgbm as lgb
        train_set = lgb.Dataset(X, y, params={'verbose': -1}, free_raw_data=False)
        if x_val is not None and y_val is not None: val_set = lgb.Dataset(x_val, y_val, params={'verbose': -1}, free_raw_data=False)
        else: val_set = None
        model = lgb.train(self.sababa_params(), train_set, num_boost_round=n_itter, early_stopping_rounds=200 if val_set is not None else None , valid_sets=[train_set] + [val_set] if val_set is not None else [], verbose_eval=100 if val_set is not None else None )
        self.model = model
        return self

    def predict(self, X): return self.model.predict(X)
    def get_params(self, deep=False): return {}