import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import metrics
from sklearn import metrics, model_selection
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import sys
reload(sys)
sys.setdefaultencoding('utf8')
# train = pd.read_csv('../train/train_featureV7.csv')
# test = pd.read_csv('../train/test_featureV7.csv')
train = pd.read_csv('../train/train_featureV0.csv')
test = pd.read_csv('../train/test_featureV0.csv')
# print np.isnan(train.drop(['uid', 'label'], axis=1).values).sum()
# print np.isnan(test.drop(['uid'], axis=1).values).sum()

importance = train.drop(['uid','label'],axis=1)
# X = train.drop(['uid', 'label'], axis=1).values
# y = train.label.values
dtrain = lgb.Dataset(train.drop(['uid','label'],axis=1),label=train.label)
dtest = lgb.Dataset(test.drop(['uid'],axis=1))


lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'is_training_metric': False,
    'min_data_in_leaf': 30,
    'num_leaves': 80,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbosity':-1
}
# lgb_params = {
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     #  'metric': ('auc', 'f1'),
#     # 'metric_freq': 100,
#     'is_training_metric': False,
#     'min_data_in_leaf': 12,
#     'num_leaves': 256,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.9,
#     'verbosity': -1,
#     #    'gpu_device_id':2,
#     #    'device':'gpu',
#     #     'lambda_l1': 0.1,
#     #    'skip_drop': 0.95,
#     #    'max_drop' : 10
#     #     'lambda_l2': 0.1
#     # 'num_threads': 18
#     #     'eta':0.07
# }
# gbm = lgb.LGBMRegressor(num_leaves=60,learning_rate=0.05, n_estimators=20)
# param_grid = {
#     'num_leaves': range(10, 80, 5),
#     'min_data_in_leaf': range(2, 20, 2),
#     'learning_rate': [0.02, 0.03, 0.04, 0.05]
# }
# gbm = GridSearchCV(gbm, param_grid,cv=4)
# gbm.fit(dtrain.data.values, dtrain.label.values)
#
# print('Best found by grid search are:', gbm.best_params_)

def evalMetric(preds, dtrain):
    label = dtrain.get_label()
    pre = pd.DataFrame({'preds': preds, 'label': label})
    pre = pre.sort_values(by='preds', ascending=False)
    auc = metrics.roc_auc_score(pre.label, pre.preds)
    pre.preds = pre.preds.map(lambda x: 1 if x >= 0.5 else 0)
    f1 = metrics.f1_score(pre.label, pre.preds)
    res = 0.6 * auc + 0.4 * f1
    # print  ('-----' % auc, f1, resf)
    return 'res', res, True

lgb.cv(lgb_params,dtrain,feval=evalMetric,early_stopping_rounds=150,verbose_eval=4,num_boost_round=10000,nfold=4,metrics=['evalMetric'])

# res_cv = lgb.cv(lgb_params, dtrain, feval=evalMetric, early_stopping_rounds=100, verbose_eval=10,
#                 num_boost_round=10000, nfold=3, metrics=['evalMetric', 'auc'], seed=1000)
model =lgb.train(lgb_params,dtrain,feval=evalMetric,verbose_eval=False,num_boost_round=400,valid_sets=[dtrain])
#
f_imp = pd.DataFrame({'feature': importance.columns.values.tolist(), 'importance': model.feature_importance()})
f_imp = f_imp.sort_values(by='importance', ascending=False)
# f_imp = f_imp[0:10]
f_imp.to_csv('./importance_features.csv', index=False, columns=['feature','importance'])
f_imp = pd.read_csv('./importance_features.csv')
# plt.xlabel('importance')
# plt.title('feature importance')
# sns.set_color_codes("muted")
# sns.barplot(x='importance', y='feature', data=f_imp, color="b")
# plt.savefig("importance.png")
# plt.show()
# for i in range(len(importance.columns.values.tolist())):
#     print importance.columns.values.tolist()[i], model.feature_importance()[i]
pred = model.predict(test.drop(['uid'],axis=1))
res =pd.DataFrame({'uid':test.uid,'label':pred})
res = res.sort_values(by='label',ascending=False)
res.label = res.label.map(lambda x: 1 if x>=0.5 else 0)
# res.to_csv('./lgb-baseline7.csv',index=False,header=False,sep=',',columns=['uid','label'])
res.to_csv('./lgb-baseline9.csv',index=False,header=False,sep=',',columns=['uid','label'])