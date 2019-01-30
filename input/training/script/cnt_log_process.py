import numpy as np
import pandas as pd
import os 


"""
因为数据集中的大多数特性都是分类的特点是，我们计算每一个的不同值的数量分类特征如表3所示。用户数，歌曲数，
艺术家的名字，作曲家，抒情诗人都很大，所以很热门编码将不是合适的预处理方法，因为维度的诅咒和记忆的代价
计算。在这种情况下，选择其他编码方法或者找到这些特性的一些表示将有助于模型。
"""

os.chdir("/home/winit/lta/Codes-for-WSDM-CUP-Music-Rec-1st-place-solution-master/input/training/script")
train = pd.read_csv('../temporal_data/train_id.csv')
test = pd.read_csv('../temporal_data/test_id.csv')
member = pd.read_csv('../temporal_data/members_id.csv')
song_origin = pd.read_csv('../temporal_data/songs_id.csv')
song_extra = pd.read_csv('../temporal_data/songs_extra_id.csv')

song = pd.DataFrame({'song_id': range(max(train.song_id.max(), test.song_id.max())+1)})
song = song.merge(song_origin, on='song_id', how='left')
song = song.merge(song_extra, on='song_id', how='left')

data = train[['msno', 'song_id']].append(test[['msno', 'song_id']])

## member_cnt  統計給每個用戶推薦的歌曲總數
mem_rec_cnt = data.groupby(by='msno').count()['song_id'].to_dict()
member['msno_rec_cnt'] = member['msno'].apply(lambda x: mem_rec_cnt[x])


####年齡大於75 或小於0的 賦值爲np.nan
member['bd'] = member['bd'].apply(lambda x: np.nan if x <= 0 or x >= 75 else x)

## song_cnt 統計藝人歌曲總數
artist_song_cnt = song.groupby(by='artist_name').count()['song_id'].to_dict()
song['artist_song_cnt'] = song['artist_name'].apply(lambda x: artist_song_cnt[x] if not np.isnan(x) else np.nan)


##統計每個作曲家的創作曲總數
composer_song_cnt = song.groupby(by='composer').count()['song_id'].to_dict()
composer_song_cnt[0] = np.nan
song['composer_song_cnt'] = song['composer'].apply(lambda x: composer_song_cnt[x] if not np.isnan(x) else np.nan)
##計每個作詞家的創作曲總數
lyricist_song_cnt = song.groupby(by='lyricist').count()['song_id'].to_dict()
lyricist_song_cnt[0] = np.nan
song['lyricist_song_cnt'] = song['lyricist'].apply(lambda x: lyricist_song_cnt[x] if not np.isnan(x) else np.nan)


#####統計第一次創作類似的歌曲總數
genre_song_cnt = song.groupby(by='first_genre_id').count()['song_id'].to_dict()
genre_song_cnt[0] = np.nan
song['genre_song_cnt'] = song['first_genre_id'].apply(lambda x: genre_song_cnt[x] if not np.isnan(x) else np.nan)

data = data.merge(song, on='song_id', how='left')


##統計一首歌推薦過給多少用戶
song_rec_cnt = data.groupby(by='song_id').count()['msno'].to_dict()
song['song_rec_cnt'] = song['song_id'].apply(lambda x: song_rec_cnt[x] if not np.isnan(x) else np.nan)



##統計一個藝術家（唱歌的）的歌推薦過給多少用戶
artist_rec_cnt = data.groupby(by='artist_name').count()['msno'].to_dict()
song['artist_rec_cnt'] = song['artist_name'].apply(lambda x: artist_rec_cnt[x] if not np.isnan(x) else np.nan)
##統計一個作曲家的歌推薦過給多少用戶
composer_rec_cnt = data.groupby(by='composer').count()['msno'].to_dict()
composer_rec_cnt[0] = np.nan
song['composer_rec_cnt'] = song['composer'].apply(lambda x: composer_rec_cnt[x] if not np.isnan(x) else np.nan)
##統計一個作詞家的歌推薦過給多少用戶
lyricist_rec_cnt = data.groupby(by='lyricist').count()['msno'].to_dict()
lyricist_rec_cnt[0] = np.nan
song['lyricist_rec_cnt'] = song['lyricist'].apply(lambda x: lyricist_rec_cnt[x] if not np.isnan(x) else np.nan)


#### 統計歌曲第一類型的，推送給用戶的總數
genre_rec_cnt = data.groupby(by='first_genre_id').count()['msno'].to_dict()
genre_rec_cnt[0] = np.nan
song['genre_rec_cnt'] = song['first_genre_id'].apply(lambda x: genre_rec_cnt[x] if not np.isnan(x) else np.nan)

## msno context features   用戶信息成分  one-hot 編碼
dummy_feat = ['source_system_tab', 'source_screen_name', 'source_type']
concat = train.drop('target', axis=1).append(test.drop('id', axis=1))

for feat in dummy_feat:
    feat_dummies = pd.get_dummies(concat[feat])
    feat_dummies.columns = ['msno_%s_'%feat + '%s'%col for col in feat_dummies.columns]
    feat_dummies['msno'] = concat['msno'].values
    feat_dummies = feat_dummies.groupby('msno').mean()
    feat_dummies['msno'] = feat_dummies.index
    member = member.merge(feat_dummies, on='msno', how='left')

train_temp = train.merge(member, on='msno', how='left')
test_temp = test.merge(member, on='msno', how='left')
###one-hot 編碼之後，取僅僅不爲0的列，作爲概率
train['msno_source_system_tab_prob'] = train_temp[[col for col in train_temp.columns if 'source_system_tab' in col]].apply(lambda x: \
        x['msno_source_system_tab_%d'%x['source_system_tab']], axis=1)
test['msno_source_system_tab_prob'] = test_temp[[col for col in test_temp.columns if 'source_system_tab' in col]].apply(lambda x: \
        x['msno_source_system_tab_%d'%x['source_system_tab']], axis=1)

train['msno_source_screen_name_prob'] = train_temp[[col for col in train_temp.columns if 'source_screen_name' in col]].apply(lambda x: \
        x['msno_source_screen_name_%d'%x['source_screen_name']], axis=1)
test['msno_source_screen_name_prob'] = test_temp[[col for col in test_temp.columns if 'source_screen_name' in col]].apply(lambda x: \
        x['msno_source_screen_name_%d'%x['source_screen_name']], axis=1)

train['msno_source_type_prob'] = train_temp[[col for col in train_temp.columns if 'source_type' in col]].apply(lambda x: \
        x['msno_source_type_%d'%x['source_type']], axis=1)
test['msno_source_type_prob'] = test_temp[[col for col in test_temp.columns if 'source_type' in col]].apply(lambda x: \
        x['msno_source_type_%d'%x['source_type']], axis=1)

## to_csv
features = ['msno_rec_cnt']
for feat in features:
    member[feat] = np.log1p(member[feat])
member.to_csv('../temporal_data/members_id_cnt.csv', index=False)

features = ['song_length', 'song_rec_cnt', 'artist_song_cnt', 'composer_song_cnt', \
        'lyricist_song_cnt', 'genre_song_cnt', 'artist_rec_cnt', \
        'composer_rec_cnt', 'lyricist_rec_cnt', 'genre_rec_cnt']
for feat in features:
    song[feat] = np.log1p(song[feat])
#song['song_length'] = np.log1p(song['song_length'])
song.to_csv('../temporal_data/songs_id_cnt.csv', index=False)

train.to_csv('../temporal_data/train_id_cnt.csv', index=False)
test.to_csv('../temporal_data/test_id_cnt.csv', index=False)
