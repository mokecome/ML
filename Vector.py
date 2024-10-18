# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:59:12 2023

@author: mokecome
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
from config import DATA_CLEAN,LABEL
class Vectorizer:
    def __init__(self):
        self.label_encoder = LabelEncoder()
              
    def tfidf(self,df_all_clean,tf_path,df_word,analyzer="word", max_features=100000, ngram_range=(1, 5)): #產生向量
    
        if analyzer=='char':
            tfidf = TfidfVectorizer(analyzer=analyzer,ngram_range=ngram_range,max_features=max_features)
        elif  analyzer=='word':  
            word_list=[]
            for row in df_all_clean: 
                word_list.extend(str(row).split(' '))
            dict_vocabulary={i:idx for idx,i in enumerate(list(dict.fromkeys(word_list+df_word.to_list())))} #特徵字
            tfidf = TfidfVectorizer(analyzer=analyzer,ngram_range=ngram_range,vocabulary=dict_vocabulary)
       
        tfidf=tfidf.fit(df_all_clean)
        pickle.dump(tfidf, open(tf_path, "wb"))
        
        if tfidf==None:
            raise ValueError('沒有產生向量')
    def get_vector(self,df,method,tf_path):      #取得向量化 data,self.tfidf    
        if method=='tfidf': 
            tf_load = pickle.load(open(tf_path, 'rb'))       
            
            feature=tf_load.transform(df[DATA_CLEAN].values.astype('U'))# 先fit再transform
            get_label_y=self.label_encoder.fit_transform(df[LABEL].values)  
                
            return feature,get_label_y
        #elif self.method=='Bert':

    
if __name__ == '__main__':
    all_data=pd.DataFrame([])
    for CATEGORY_LABEL in ['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸','乳品','蛋品','肉類','海鮮','水果','蔬菜','冰品','米油雜糧調味料','泡麵罐頭調理','巧克力','酒類','棉、紙製品','母嬰用品','居家生活','服飾鞋包','保養美妝','3C家電','保健生機','盥洗用品','禮盒伴手禮','醫療護理','18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']:
        try:
            train_x=pd.read_excel(f'{CATEGORY_LABEL}_train_x.xlsx').replace(np.nan,'')
            test_x=pd.read_excel(f'{CATEGORY_LABEL}_test_x.xlsx').replace(np.nan,'')
        except:
            pass
        all_data=pd.concat([all_data,train_x,test_x],axis=0)
    all_data=all_data.drop_duplicates()   
    
    vector=Vectorizer(all_data[DATA_CLEAN],'tfidf',tf_path="utils/tfidf_feature_all.pkl")