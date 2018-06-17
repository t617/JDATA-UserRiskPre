# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

uid_train = pd.read_csv('../train/uid_train.txt',sep='\t',header=None,names=('uid','label'))
voice_train = pd.read_csv('../train/voice_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'), dtype={'start_time':np.str,'end_time':np.str})
sms_train = pd.read_csv('../train/sms_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_train = pd.read_csv('../train/wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':np.str})
# wa_train = pd.read_csv('../train/wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'))

voice_test = pd.read_csv('../train/voice_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'), dtype={'start_time':np.str,'end_time':np.str})
sms_test = pd.read_csv('../train/sms_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':np.str})
wa_test = pd.read_csv('../train/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':np.str})
# wa_test = pd.read_csv('../train/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'))
# voice_test_a = pd.read_csv('../train/voice_test_a.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str},encoding='utf-8')
# sms_test_a = pd.read_csv('../train/sms_test_a.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str},encoding='utf-8')
# wa_test_a = pd.read_csv('../train/wa_test_a.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str},encoding='utf-8')

uid_test = pd.DataFrame({'uid':pd.unique(wa_test['uid'])})
uid_test.to_csv('../train/uid_test_b.txt',index=None)

# voice = pd.concat([voice_train,voice_test_a,voice_test],axis=0)
# sms = pd.concat([sms_train,sms_test_a,sms_test],axis=0)
# wa = pd.concat([wa_train,wa_test_a,wa_test],axis=0)

voice = pd.concat([voice_train,voice_test],axis=0)
sms = pd.concat([sms_train,sms_test],axis=0)
wa = pd.concat([wa_train,wa_test],axis=0)

voice.drop_duplicates(inplace=True)
sms.drop_duplicates(inplace=True)
wa.drop_duplicates(inplace=True)
# print np.isnan(voice[['opp_len','call_type','in_out']].values).sum()
# print np.isnan(sms[['opp_len','in_out']].values).sum()
# print np.isnan(wa['down_flow'].values).sum()
# print np.isnan(wa['up_flow'].values).sum()
# print np.isnan(wa_train['wa_type'].values).sum()
# print np.isnan(wa_test['wa_type'].values).sum()

# print np.isnan(wa[['visit_cnt','up_flow','visit_dura','down_flow','wa_type']].values).sum()

voice_in_out = voice.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)

voice_in_out['voice_in_out-mean']=voice_in_out.voice_in_out_1 - np.mean(voice_in_out.voice_in_out_1)
voice_in_out_unique = voice.groupby(['uid','in_out'])['opp_num'].nunique().unstack().add_prefix('voice_in_out_unique_').reset_index().fillna(0)
voice_in_out_unique['voice_in_out_unique-mean']=voice_in_out_unique.voice_in_out_unique_1 - np.mean(voice_in_out_unique.voice_in_out_unique_1)
voice_in_out['voice_in_out_diff'] = voice_in_out.voice_in_out_1 - voice_in_out.voice_in_out_0
# voice_in_out = voice_in_out[voice_in_out.voice_in_out_0 + voice_in_out.voice_in_out_1 > 0]
# voice_in_out['voice_in_out_rate_of_1'] = (voice_in_out.voice_in_out_1 / (voice_in_out.voice_in_out_1 + voice_in_out.voice_in_out_0)).astype('float')
# voice_in_out['voice_in_out_rate_of_0'] = (voice_in_out.voice_in_out_0 / (voice_in_out.voice_in_out_1 + voice_in_out.voice_in_out_0)).astype('float')
voice_in_out_unique['voice_in_out_unique_diff'] = voice_in_out_unique.voice_in_out_unique_1 - voice_in_out_unique.voice_in_out_unique_0
# voice_in_out_unique = voice_in_out_unique[voice_in_out_unique.voice_in_out_unique_0 + voice_in_out_unique.voice_in_out_unique_1 > 0]
# voice_in_out_unique['voice_in_out_unique_rate_of_1'] = (voice_in_out_unique.voice_in_out_unique_1 / (voice_in_out_unique.voice_in_out_unique_1 + voice_in_out_unique.voice_in_out_unique_0)).astype('float')
# voice_in_out_unique['voice_in_out_unique_rate_of_0'] = (voice_in_out_unique.voice_in_out_unique_0 / (voice_in_out_unique.voice_in_out_unique_1 + voice_in_out_unique.voice_in_out_unique_0)).astype('float')

# voice = voice[voice.in_out==1]
voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_').reset_index().fillna(0)
# voice_opp_num.voice_opp_num_count = voice_opp_num.voice_opp_num_count.map(lambda x: 5000 if x>=5000 else x)
# voice_opp_num = voice_opp_num[voice_opp_num.voice_opp_num_count > 0]
# voice_opp_num['voice_unique_in_count'] = (voice_opp_num.voice_opp_num_unique_count / voice_opp_num.voice_opp_num_count).astype('float')
voice_opp_num['voice_count-mean'] = voice_opp_num.voice_opp_num_count - np.mean(voice_opp_num.voice_opp_num_count)
voice_opp_num['voice_unique_count-mean'] = voice_opp_num.voice_opp_num_unique_count - np.mean(voice_opp_num.voice_opp_num_unique_count)

voice_opp_head_17_count = voice.groupby(['uid'])['opp_head'].agg({'17_': lambda x: np.sum(x.values == 170) + np.sum(x.values == 171)}).add_prefix('voice_opp_head_').reset_index().fillna(0)
# voice_head_and_call_type_count = voice.groupby(['uid','call_type'])['opp_head'].nunique().unstack().add_prefix('voice_opp_head_type_').reset_index().fillna(0)
# voice_17_and_type_3_count = voice.groupby(['uid','call_type'])['opp_head'].agg({'3_count': lambda x: np.sum(x.values == 170) + np.sum(x.values == 171)}).unstack().add_prefix('voice_opp_head_17_type_').reset_index().fillna(0)
# voice_opp_head_17_count['voice_rate_of_17_'] = (voice_opp_head_17_count.voice_opp_head_17_ / voice_opp_num.voice_opp_num_count).astype('float')
voice_opp_head_100_count = voice.groupby(['uid'])['opp_head'].agg({'100': lambda x: np.sum(x.values == 1)}).add_prefix('voice_opp_head_').reset_index().fillna(0)
voice_opp_head_100_count['voice_head_100_count-mean'] = voice_opp_head_100_count.voice_opp_head_100 - np.mean(voice_opp_head_100_count.voice_opp_head_100)
voice_in_out_head_100 = voice.groupby(['uid','in_out'])['opp_head'].agg({'100': lambda x: np.sum(x.values == 1)}).unstack().add_prefix('voice_in_out_head_').reset_index().fillna(0)

# voice_opp_head_100_count['voice_rate_of_100'] = (voice_opp_head_100_count.voice_opp_head_100 / voice_opp_num.voice_opp_num_count).astype('float')
voice_opp_head_100_count['voice_head_of_not_100'] = voice_opp_num.voice_opp_num_count - voice_opp_head_100_count.voice_opp_head_100
# voice_opp_head_100_count['voice_rate_of_not_100'] = 1 - voice_opp_head_100_count.voice_rate_of_100

voice_opp_head = voice.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_').reset_index().fillna(0)
voice_opp_len_type = voice.groupby(['uid'])['opp_len'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_len_type_').reset_index().fillna(0)

voice.opp_len = voice.opp_len.map(lambda x: 0 if (x==3 or x==6 or x == 14 or x>15) else x)
voice_opp_len=voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)
voice.call_type = voice.call_type.map(lambda x: 0 if (x > 3) else x)
voice_call_type = voice.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index()
voice_call_type_unique = voice.groupby(['uid','call_type'])['opp_num'].nunique().unstack().add_prefix('voice_call_type_unique_').reset_index()

voice['voice_dura_time'] = ((voice.end_time.str.slice(2, 4).astype(int) - voice.start_time.str.slice(2, 4).astype(int))* 3600 + (voice.end_time.str.slice(4, 6).astype(int) - voice.start_time.str.slice(4, 6).astype(int)) * 60 + voice.end_time.str.slice(6, 8).astype(int) - voice.start_time.str.slice(6, 8).astype(int))
voice_dura_time = voice.groupby(['uid'])['voice_dura_time'].agg(['std','max','min','median','mean','sum']).add_prefix('voice_dura_time_').reset_index().fillna(0)
voice_in_out_dura_time = voice.groupby(['uid','in_out'])['voice_dura_time'].sum().unstack().add_prefix('voice_in_out_dura_time_').reset_index().fillna(0)
voice['voice_start_hour_time'] = ((voice.start_time.str.slice(2, 4).astype('int') - 1) / 8).astype('int')
voice_start_time = voice.groupby(['uid','voice_start_hour_time'])['uid'].count().unstack().add_prefix('voice_start_time_').reset_index().fillna(0)
# voice['date'] = voice.start_time.str.slice(0, 2).astype('int')
voice['date'] = ((voice.start_time.str.slice(0, 2).astype('int') - 1)/ 5).astype('int')
voice_date = voice.groupby(['uid','date'])['uid'].count().unstack().add_prefix('voice_date_').reset_index().fillna(0)
voice_date_unique = voice.groupby(['uid','date'])['opp_num'].nunique().unstack().add_prefix('voice_date_').reset_index().fillna(0)



# sms_opp_num_has_0 = sms.groupby(['uid'])['opp_num'].agg({'count':'count'}).add_prefix('sms_opp_num_has_0_').reset_index().fillna(0)
sms_opp_head_0_count =sms.groupby(['uid'])['opp_head'].agg({'0': lambda x: np.sum(x.values == 0)}).add_prefix('sms_opp_head_').reset_index().fillna(0)
# sms_opp_head_0_count['sms_rate_of_not_0'] = ((sms_opp_num_has_0.sms_opp_num_has_0_count - sms_opp_head_0_count.sms_opp_head_0) / sms_opp_num_has_0.sms_opp_num_has_0_count).astype('float')
sms_opp_head_0_count['sms_head_0_count-mean'] = sms_opp_head_0_count.sms_opp_head_0 - np.mean(sms_opp_head_0_count.sms_opp_head_0)
sms = sms[sms.opp_head != 0]

sms_in_out = sms.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)
sms_in_out['sms_in_out-mean'] = sms_in_out.sms_in_out_1 - np.mean(sms_in_out.sms_in_out_1)
sms_in_out_unique = sms.groupby(['uid','in_out'])['opp_num'].nunique().unstack().add_prefix('sms_in_out_unique_').reset_index().fillna(0)
sms_in_out_unique['sms_in_out_unique-mean'] = sms_in_out_unique.sms_in_out_unique_1 - np.mean(sms_in_out_unique.sms_in_out_unique_1)

# sms_in_out = sms_in_out[sms_in_out.sms_in_out_0 + sms_in_out.sms_in_out_1 > 0]
# sms_in_out_unique = sms_in_out_unique[sms_in_out_unique.sms_in_out_unique_0 + sms_in_out_unique.sms_in_out_unique_1 > 0]
# sms_in_out['sms_in_in_out_count'] = (sms_in_out.sms_in_out_1 / (sms_in_out.sms_in_out_1 + sms_in_out.sms_in_out_0)).astype('float')
# sms_in_out_unique['sms_in_in_out_count'] = (sms_in_out_unique.sms_in_out_unique_1 / (sms_in_out_unique.sms_in_out_unique_1 + sms_in_out_unique.sms_in_out_unique_0)).astype('float')

# sms = sms[sms.in_out == 1]
sms_opp_num = sms.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_').reset_index().fillna(0)
# sms_opp_num = sms_opp_num[sms_opp_num.sms_opp_num_count > 0]
# sms_opp_num['sms_unique_in_count'] = (sms_opp_num.sms_opp_num_unique_count / sms_opp_num.sms_opp_num_count).astype('float')
sms_opp_num['sms_count-mean'] = (sms_opp_num.sms_opp_num_count - np.mean(sms_opp_num.sms_opp_num_count)).astype('float')
sms_opp_num['sms_unique_count-mean'] = (sms_opp_num.sms_opp_num_unique_count - np.mean(sms_opp_num.sms_opp_num_unique_count)).astype('float')
sms_opp_num['sms_opp_num_diff']=sms_opp_num.sms_opp_num_count - sms_opp_num.sms_opp_num_unique_count
sms_opp_head=sms.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_').reset_index().fillna(0)
# sms.opp_len = sms.opp_len.map(lambda x: -1 if (x < 7 or x>14) else x)
sms.opp_len = sms.opp_len.map(lambda x: -1 if (x==3 or x==6 or x>15) else x)
sms_opp_len = sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)
sms_opp_len_type = sms.groupby(['uid'])['opp_len'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_len_type_').reset_index().fillna(0)

sms['sms_start_hour_time'] = (sms.start_time.str.slice(2, 4).astype('int') / 3).astype('int')
# sms.sms_start_hour_time = sms.sms_start_hour_time.map(lambda x: 0 if ((x < 24 and x > 17) or (x>= 0 and x < 8)) else x)
sms_start_time = sms.groupby(['uid','sms_start_hour_time'])['uid'].count().unstack().add_prefix('sms_start_time_').reset_index().fillna(0)

sms_opp_head_17_count=sms.groupby(['uid'])['opp_head'].agg({'17_': lambda x: np.sum(x.values == 170) + np.sum(x.values == 171)}).add_prefix('sms_opp_head_').reset_index().fillna(0)
# sms_opp_head_17_count=sms.groupby(['uid'])['opp_head'].agg({'17_': lambda x: np.exp(np.sum(x.values == 170) + np.sum(x.values == 171))}).add_prefix('sms_opp_head_').reset_index().fillna(0)
# sms_opp_head_17_count['sms_rate_of_17_'] = (sms_opp_head_17_count.sms_opp_head_17_ / sms_opp_num.sms_opp_num_count).astype('float')
sms_opp_head_106_count =sms.groupby(['uid'])['opp_head'].agg({'106': lambda x: np.sum(x.values == 106)}).add_prefix('sms_opp_head_').reset_index().fillna(0)
# sms_opp_head_106_unique_count = sms.groupby(['uid'])[['opp_head','opp_num']].agg({'unique_106': lambda x: len(pd.unique(x.opp_head.values==106))}).add_prefix('sms_opp_head_').reset_index().fillna(0)

sms_opp_head_106_count['sms_opp_head_106_count-mean'] = sms_opp_head_106_count.sms_opp_head_106 - np.mean(sms_opp_head_106_count.sms_opp_head_106)
# sms_opp_head_106_count['sms_rate_of_106'] = (sms_opp_head_106_count.sms_opp_head_106 / sms_opp_num.sms_opp_num_count).astype('float')
sms_opp_head_106_count['sms_opp_head_not_106_count'] = sms_opp_num.sms_opp_num_count - sms_opp_head_106_count.sms_opp_head_106
sms_opp_head_106_count['sms_opp_head_not_106_count-mean'] = sms_opp_head_106_count.sms_opp_head_not_106_count - np.mean(sms_opp_head_106_count.sms_opp_head_not_106_count)
# sms_opp_head_106_count['sms_rate_of_not_106'] = (sms_opp_num.sms_opp_num_count - sms_opp_head_106_count.sms_opp_head_106 / sms_opp_num.sms_opp_num_count).astype('float')

sms_opp_head_100_count =sms.groupby(['uid'])['opp_head'].agg({'100': lambda x: np.sum(x.values == 1)}).add_prefix('sms_opp_head_').reset_index().fillna(0)
sms_opp_head_100_count['sms_opp_head_100_count-mean'] = sms_opp_head_100_count.sms_opp_head_100 - np.mean(sms_opp_head_100_count.sms_opp_head_100)
# sms_opp_head_100_count['sms_rate_of_100'] = (sms_opp_head_100_count.sms_opp_head_100 / sms_opp_num.sms_opp_num_count).astype('float')
sms_opp_head_100_count['sms_opp_head_not_100_count'] = sms_opp_num.sms_opp_num_count - sms_opp_head_100_count.sms_opp_head_100
sms_opp_head_100_count['sms_opp_head_not_100_count-mean'] = sms_opp_head_100_count.sms_opp_head_not_100_count - np.mean(sms_opp_head_100_count.sms_opp_head_not_100_count)
# sms_opp_head_100_count['sms_rate_of_not_100'] = 1 - sms_opp_head_100_count.sms_rate_of_100
# sms_opp_head_100_count['sms_rate_of_not_100_and_106'] = 1 - sms_opp_head_100_count.sms_rate_of_100 - sms_opp_head_106_count.sms_rate_of_106

# sms['sms_date'] = sms.start_time.str.slice(0, 2).astype('int')
sms['sms_date'] = ((sms.start_time.str.slice(0, 2).astype('int')-1) / 5).astype('int')
sms_date_count = sms.groupby(['uid', 'sms_date'])['uid'].count().unstack().add_prefix('sms_date_').reset_index().fillna(0)
sms_date_count_unique = sms.groupby(['uid', 'sms_date'])['opp_num'].nunique().unstack().add_prefix('sms_date_unique_').reset_index().fillna(0)



# wa.date = wa.date.str.slice(0, 2).fillna('0').astype('int')
# wa.date = ((wa.date.str.slice(0, 2).fillna(str(1 + np.random.randint(45))).astype('int') - 1) / 5).astype('int')
wa.date = ((wa.date.fillna('-4').astype('int') - 1)/5).astype('int')
wa_date_count = wa.groupby(['uid', 'date'])['uid'].count().unstack().reset_index()
wa_date_count_unique = wa.groupby(['uid', 'date'])['wa_name'].nunique().unstack().reset_index()

wa_name = wa.groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('wa_name_').reset_index().fillna(0)
wa_name['wa_name_count-unique'] = wa_name.wa_name_count - wa_name.wa_name_unique_count
wa_name['wa_name_count-mean'] = wa_name.wa_name_count - np.mean(wa_name.wa_name_count)
wa_name['wa_name_unique_count-mean'] = wa_name.wa_name_unique_count - np.mean(wa_name.wa_name_unique_count)
# wa_name = wa_name[wa_name.wa_name_count > 0]
# wa_name['wa_unique_in_count'] = (wa_name.wa_name_unique_count / wa_name.wa_name_count).astype('float')
# wa_name_tb = wa.groupby(['uid'])['wa_name'].agg({'tb': lambda x: np.sum(x.values == '手机淘宝') + np.sum(x.values == '淘宝网')+ np.sum(x.values == '天猫商城') + np.sum(x.values == '天猫') + np.sum(x.values == '今日头条')+ np.sum(x.values == '快手')}).reset_index()
wa['wa_name_len'] = wa.wa_name.str.len() / 4
wa.wa_name_len = wa.wa_name_len.map(lambda x: -1 if (x > 10) else x)
wa_name_len = wa.groupby(['uid', 'wa_name_len'])['uid'].count().unstack().add_prefix('wa_name_len_').reset_index().fillna(0)
visit_cnt = wa.groupby(['uid'])['visit_cnt'].agg(['std','max','median','mean','sum']).add_prefix('wa_visit_cnt_').reset_index().fillna(0)
# wa.visit_dura = wa.visit_dura.map(lambda x: 86400 if (x > 86400) else x)
wa.visit_dura = wa.visit_dura/60
visit_dura = wa.groupby(['uid'])['visit_dura'].agg(['std','max','median','mean','sum']).add_prefix('wa_visit_dura_').reset_index().fillna(0)
# wa.up_flow = np.log2(1+wa.up_flow/1024)
wa.up_flow = wa.up_flow/1024
up_flow = wa.groupby(['uid'])['up_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_up_flow_').reset_index().fillna(0)
# wa.down_flow = np.log2(1+wa.down_flow/1024)
wa.down_flow = wa.down_flow/1024
down_flow = wa.groupby(['uid'])['down_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_down_flow_').reset_index().fillna(0)

wa_type = wa.groupby(['uid','wa_type'])['uid'].count().unstack().add_prefix('wa_type_').reset_index().fillna(0)
wa_type_unique = wa.groupby(['uid','wa_type'])['wa_name'].nunique().unstack().add_prefix('wa_type_unique_').reset_index().fillna(0)

# print np.isnan(wa.date).sum()
# wa.date = wa.date.fillna(wa.date.median()).astype('int')

wa_date_up_flow = wa.groupby(['uid', 'date'])['up_flow'].agg(['std','max','min','median','mean','sum']).unstack().add_prefix('wa_date_up_').reset_index().fillna(0)
wa_date_down_flow = wa.groupby(['uid', 'date'])['down_flow'].agg(['std','max','min','median','mean','sum']).unstack().add_prefix('wa_date_down_').reset_index().fillna(0)
wa_date_visit_cnt = wa.groupby(['uid', 'date'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).unstack().add_prefix('wa_date_visit_cnt_').reset_index().fillna(0)
# wa_date_visit_dura = wa.groupby(['uid', 'date'])['visit_dura'].agg(['std','max','min','median','mean','sum']).unstack().add_prefix('wa_date_visit_dura_').reset_index().fillna(0)

# wa_date_up_flow = wa.groupby(['uid', 'date'])['up_flow'].sum().unstack().add_prefix('wa_date_up_').reset_index().fillna(0)
# wa_date_visit_cnt = wa.groupby(['uid', 'date'])['visit_cnt'].sum().unstack().add_prefix('wa_date_visit_cnt_').reset_index().fillna(0)
# wa_date_down_flow = wa.groupby(['uid', 'date'])['down_flow'].sum().unstack().add_prefix('wa_date_down_').reset_index().fillna(0)
# wa_date_visit_dura = wa.groupby(['uid', 'date'])['visit_dura'].sum().unstack().add_prefix('wa_date_visit_dura_').reset_index().fillna(0)

wa_type_up_flow = wa.groupby(['uid', 'wa_type'])['up_flow'].agg(['std','max','min','median','mean','sum']).unstack().add_prefix('wa_type_up_').reset_index().fillna(0)
wa_type_down_flow = wa.groupby(['uid', 'wa_type'])['down_flow'].agg(['std','max','min','median','mean','sum']).unstack().add_prefix('wa_type_down_').reset_index().fillna(0)
wa_type_visit_cnt = wa.groupby(['uid', 'wa_type'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).unstack().add_prefix('wa_type_visit_cnt_').reset_index().fillna(0)
# wa_type_visit_dura = wa.groupby(['uid', 'wa_type'])['visit_dura'].agg(['std','max','min','median','mean','sum']).unstack().add_prefix('wa_type_visit_dura_').reset_index().fillna(0)
# wa_type_up_flow = wa.groupby(['uid', 'wa_type'])['up_flow'].sum().unstack().add_prefix('wa_type_up_').reset_index().fillna(0)
# wa_type_visit_cnt = wa.groupby(['uid', 'wa_type'])['visit_cnt'].sum().unstack().add_prefix('wa_type_visit_cnt_').reset_index().fillna(0)
# wa_type_down_flow = wa.groupby(['uid', 'wa_type'])['down_flow'].sum().unstack().add_prefix('wa_type_down_').reset_index().fillna(0)
# wa_type_visit_dura = wa.groupby(['uid', 'wa_type'])['visit_dura'].sum().unstack().add_prefix('wa_type_visit_dura_').reset_index().fillna(0)
# voice_in_out_type = voice.groupby(['uid','call_type'])['in_out'].agg({'in_out_0': lambda x: np.sum(x==0),'in_out_1': lambda x: np.sum(x==1)}).unstack().add_prefix('voice_in_out_type_').reset_index().fillna(0)
# voice_in_out_type_unique = voice.groupby(['uid','in_out','call_type'])['opp_num'].nunique().unstack().add_prefix('voice_in_out_type_unique_').reset_index().fillna(0)
# sms_in_out_len = sms.groupby(['uid','opp_len'])['in_out'].agg({'in_out_0': lambda x: np.sum(x==0),'in_out_1': lambda x: np.sum(x==1)}).unstack().add_prefix('sms_in_out_len_').reset_index().fillna(0)
# sms_in_out_len_unique = sms.groupby(['uid','in_out','opp_len'])['opp_num'].nunique().unstack().add_prefix('sms_in_out_len_unique_').reset_index().fillna(0)
# wa_date_type = wa.groupby(['uid','date'])['wa_type'].agg({'wa_type_0': lambda x: np.sum(x==0),'wa_type_1': lambda x: np.sum(x==1)}).unstack().add_prefix('wa_date_type_').reset_index().fillna(0)
# wa_date_type_unique = wa.groupby(['uid','date','wa_type'])['wa_name'].nunique().unstack().add_prefix('wa_date_type_unique_').reset_index().fillna(0)
def time_gap(start, end):
    if pd.isnull(start):
        return np.nan
    end_day = int(str(end)[0:2])
    start_day = int(str(start)[0:2])
    day_gap = (end_day - start_day) * 86400

    end_hour = int(str(end)[2:4])
    start_hour = int(str(start)[2:4])
    hour_gap = (end_hour - start_hour) * 3600

    end_min = int(str(end)[4:6])
    start_min = int(str(start)[4:6])
    min_gap = (end_min - start_min) * 60

    end_sec = int(str(end)[6:8])
    start_sec = int(str(start)[6:8])
    sec_gap = (end_sec - start_sec)

    return day_gap + hour_gap + min_gap + sec_gap
voice_sort = (voice.sort_values(by=['start_time','end_time'],ascending=True)).reset_index()
voice_sort['last_end_time']=voice_sort.groupby(['uid'])['end_time'].apply(lambda i:i.shift(1))
voice_sort['last_gap_time'] = voice_sort[['last_end_time','start_time']].apply(lambda x: time_gap(x[0],x[1]),axis=1)
voice_last_gap_time=voice_sort.groupby(['uid'])['last_gap_time'].agg(['std','max','min','median','mean','sum',np.ptp]).add_prefix('voice_last_gap_time_').reset_index()
sms_sort = sms.sort_values(by=['uid','start_time'],ascending='True').reset_index()
sms_sort['last_start_time']=sms_sort.groupby(['uid'])['start_time'].apply(lambda i:i.shift(1))
sms_sort['last_start_gap_time'] = sms_sort[['last_start_time','start_time']].apply(lambda x: time_gap(x[0],x[1]),axis=1)
sms_last_start_gap_time=sms_sort.groupby(['uid'])['last_start_gap_time'].agg(['std','max','min','median','mean','sum',np.ptp]).add_prefix('sms_last_start_gap_time_').reset_index()

opp_num_list = voice.groupby(['opp_num'])['uid'].count().sort_values(ascending=False).reset_index()['opp_num'][0:1000].values
voice_each_opp_num_count=voice[voice.opp_num.map(lambda x: x in opp_num_list)].groupby(['uid','opp_num'])['uid'].count().unstack().add_prefix('voice_each_opp_num_count_').reset_index().fillna(0)

sms_opp_num_list = sms.groupby(['opp_num'])['uid'].count().sort_values(ascending=False).reset_index()['opp_num'][0:1000].values
sms_each_opp_num_count=sms[sms.opp_num.map(lambda x: x in sms_opp_num_list)].groupby(['uid','opp_num'])['uid'].count().unstack().add_prefix('sms_each_opp_num_count_').reset_index().fillna(0)

wa_name_list = wa.groupby(['wa_name'])['uid'].count().sort_values(ascending=False).reset_index()['wa_name'][0:1000].values
wa_each_name_count=wa[wa.wa_name.map(lambda x: x in wa_name_list)].groupby(['uid','wa_name'])['uid'].count().unstack().add_prefix('wa_each_name_count_').reset_index().fillna(0)

voice_each_opp_head_count=voice.groupby(['uid','opp_head'])['uid'].count().unstack().add_prefix('voice_each_opp_head_count_').reset_index().fillna(0)

sms_each_opp_head_count = sms.groupby(['uid', 'opp_head'])['uid'].count().unstack().add_prefix(
    'sms_each_opp_head_count_').reset_index().fillna(0)
feature = [
           # voice_in_out_type,#voice_in_out_type_unique,
           # sms_in_out_len,#sms_in_out_len_unique,
           # wa_date_type,#wa_date_type_unique,

           voice_in_out, voice_in_out_unique,
           voice_opp_num, voice_opp_head,
           # voice_opp_head_100_count, voice_in_out_head_100,#voice_opp_head_17_count,
           voice_opp_len, voice_opp_len_type, voice_start_time, voice_call_type, voice_call_type_unique,

           voice_date, voice_date_unique, voice_dura_time,voice_in_out_dura_time,# voice_17_and_type_3_count,
           # voice_head_and_call_type_count,

           sms_opp_head_0_count,
           sms_in_out, sms_in_out_unique,
           sms_opp_num, sms_opp_head, sms_opp_len,sms_opp_len_type,
           # sms_opp_head_106_count, sms_opp_head_100_count,sms_opp_head_17_count,
           sms_start_time, sms_date_count, #sms_date_count_unique,

           wa_name, wa_date_count, wa_date_count_unique, wa_name_len,
           # wa_name_tb,
           up_flow, down_flow, visit_dura, visit_cnt, wa_type,wa_type_unique,
           # wa_type_up_flow,wa_type_down_flow,wa_type_visit_cnt,#wa_type_visit_dura,
           # wa_date_up_flow, wa_date_down_flow,wa_date_visit_cnt,#wa_date_visit_dura,
           # voice_last_gap_time,sms_last_start_gap_time,
           # voice_each_opp_num_count,sms_each_opp_num_count,wa_each_name_count,
           sms_each_opp_head_count,
           voice_each_opp_head_count
           ]

train_feature = uid_train
for feat in feature:
    train_feature=pd.merge(train_feature, feat, how='left',on='uid').fillna(0)

test_feature = uid_test
for feat in feature:
    test_feature=pd.merge(test_feature,feat,how='left',on='uid').fillna(0)

# train_feature['wa_sign_and_106*'] = train_feature.sms_opp_head_106 * train_feature.wa_name_unique_count
# train_feature['voice_in_and_sms_in*'] = train_feature.voice_in_out_unique_1 * train_feature.sms_in_out_unique_1
# # train_feature['wa_and_sms_in*'] = train_feature.wa_name_unique_count * train_feature.sms_in_out_unique_1
# train_feature['wa_sms_voice*'] = train_feature.sms_in_out_unique_1 * train_feature.wa_name_unique_count * train_feature.voice_in_out_unique_1
# train_feature['wa_name_sms*'] = train_feature.wa_name_count * train_feature.sms_opp_num_count

# train_feature.to_csv('../train/train_featureV7.c sv',index=None)
# test_feature.to_csv('../train/test_featureV7.csv',index=None)
train_feature.to_csv('../train/train_featureV0.csv',index=None)
test_feature.to_csv('../train/test_featureV0.csv',index=None)