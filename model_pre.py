# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:42:51 2023

@author: mokecome
"""
import pandas as pd
import numpy as np
#from utils.metric import Metric
import pickle
import joblib
import jieba 
import jieba.posseg as pseg 
from utils.text_process import proccessing,remove_punctuations
import re
def check_english(x):
    my_re = re.compile(r'[A-Za-z]',re.S)
    res = re.findall(my_re,x)
    if len(res):
        return True
    else:
        return False
def get_maxlen_english(x):
    s_en=re.sub(r"[^a-zA-Z ]+",' ',x)
    res=max(s_en.split(' '),key=len,default='')
    return res

def get_maxlen_ennum(x):#最長英文 數字
    s_en=re.sub(r"[^a-zA-Z0-9 ]+",' ',x)
    res=max(s_en.split(' '),key=len,default='')
    return res

dict_list=pd.read_excel('utils/dict.xlsx')['詞'].tolist()
stopword_list=pd.read_excel('utils/stopwords.xlsx')['stopword'].tolist()  
with open('utils/alllist.txt','w',encoding='utf-8')as f:
      for i in dict_list: 
        f.write(str(i)+' {} {}'.format(100,'oo')+'\n')
jieba.load_userdict('utils/alllist.txt')  
def tokenizer(all_str,flag):            
    if flag==-1:#cut_all = False，保留所有1個字的斷詞結果  
       new_list=[]
       cut_list=[]
       all_str=remove_punctuations(all_str)
       cut_list=[i for i in  all_str.strip().split(' ')]        
       
       for text in cut_list:
           text_after_jieba=jieba.cut(text,cut_all=False) 
           for cut_str in text_after_jieba:#分詞後結果 
               seq_list=pseg.cut(cut_str)
               for w,p in seq_list:
                    if w in dict_list:
                       new_list.append(w) #只能是關鍵字len(w)=1
                    else:
                         if w not in stopword_list  and (p.startswith('N') or  p.startswith('A') or  p.startswith('oo')):#不是停用詞  #名詞  形容詞                            
                            if w not in new_list:
                               new_list.append(w)
       #print(' '.join(new_list))
                      
       return ' '.join(new_list)
   
    if flag==0:#cut_all = True，保留1個字的斷詞結果
       new_list=[]
       cut_list=[]
       all_str=remove_punctuations(all_str)
       cut_list=[i for i in  all_str.strip().split(' ')]        
       
       for text in cut_list:
           if len(all_str)<4:
               text_after_jieba=jieba.cut(text,cut_all=False) #中文過短
           else:
               text_after_jieba=jieba.cut(text,cut_all=True) 
               
           for cut_str in text_after_jieba:#分詞後結果 
               seq_list=pseg.cut(cut_str)
               for w,p in seq_list:
                    if w in dict_list:
                       new_list.append(w) #只能是關鍵字len(w)=1
                    else:
                         if w not in stopword_list  and (p.startswith('N') or  p.startswith('A') or  p.startswith('oo')):#不是停用詞  #名詞  形容詞
                             if w not in new_list:
                                   new_list.append(w)                          
       return ' '.join(new_list)
      
    if flag==1:
       new_list=[]
       cut_list=[]
       english_keyword_list=[i for i in  dict_list if check_english(i)]
       if any(_ in all_str for _ in english_keyword_list): #有該關鍵字
           jieba.re_han_default=re.compile('(.+)',re.U)
           text_after_jieba=jieba.cut(all_str,cut_all=False)
           for cut_str in text_after_jieba:#分詞後結果 
               seq_list=pseg.cut(cut_str)
               for w,p in seq_list:
                   if w in dict_list:
                       if len(w)<4 and w.encode('UTF-8').isalpha(): #為英文且長度小於4
                          if get_maxlen_english(all_str)==w:
                              new_list.append(w) 
                       elif len(w)<5 and w.encode('UTF-8').isalnum():#英數字
                           if get_maxlen_ennum(all_str)==w:
                               new_list.append(w) 
                                
                       else:#中英文 中文
                           new_list.append(w)
           new_list=[i for i in  new_list if check_english(i)]#只保留含有英文
          
           return ' '.join(new_list)
       else:
           return ''
       
def pl(all_str):
    cut_list=[]
    for i in  str(all_str).strip().split(' '):
        if len(i)>1 or (len(i)==1 and i in dict_list):
            cut_list.append(i) 
    return ' '.join(cut_list)       

        
   



#檢查 合併create_label   df=df.data_clean()

# df.to_excel('預測烘焙甜點爬蟲資料_20230425_.xlsx')
if __name__ == '__main__': 
    #version='(cut=false)'
    version='(cut=false has1)'
    #version='(cut=true)'
    #version='(cut=true has1)'
    
    df1=pd.DataFrame([])
    filename='外部商品_20230815'
    # filename='團購分類測試data_clean'
    # df=pd.read_excel(f'../model_test/{filename}.xlsx').replace('nan','').replace(np.nan,'')  
    # df_dict=pd.read_excel('utils/dict.xlsx')['詞'].tolist()
    # df_stopword=pd.read_excel('utils/stopwords.xlsx')['stopword'].tolist()  
    
    # df['商品名稱_temp']=df['商品名稱'].apply(proccessing) 
    # if 'true' in version:
    #     df['中文']=df['商品名稱_temp'].apply(tokenizer,args=(0,))
    # else:
    #     df['中文']=df['商品名稱_temp'].apply(tokenizer,args=(-1,))
        
    # df['英文']=df['商品名稱_temp'].apply(tokenizer,args=(1,))    
    # df['data_clean']= df['中文']+' '+df['英文']
    
    # if 'has1' not in version:
    #     df['中文']=df['中文'].apply(pl)
    #     df['data_clean']=df['data_clean'].apply(pl)
    # df.to_excel(f"../model_test/{filename}{version}.xlsx", index=False)
    df=pd.read_excel(f"../model_test/{filename}{version}.xlsx")
    '''
    all_data=pd.DataFrame([])
    for CATEGORY_LABEL in ['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸','乳品','蛋品','肉類','海鮮','水果','蔬菜','冰品','米油雜糧調味料','泡麵罐頭調理','巧克力','酒類','棉、紙製品','母嬰用品','居家生活','服飾鞋包','保養美妝','3C家電','保健生機','盥洗用品','禮盒伴手禮','醫療護理','18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']:
        train_x=pd.read_excel(f'D:/分類model/大分類模型{version}/ramdom_split/{CATEGORY_LABEL}_train_x.xlsx').replace("nan",'').replace(np.nan,'')
        test_x=pd.read_excel(f'D:/分類model/大分類模型{version}/ramdom_split/{CATEGORY_LABEL}_test_x.xlsx').replace("nan",'').replace(np.nan,'')
       
        all_data=pd.concat([all_data,train_x,test_x],axis=0)
    
    all_data[[DATA_CLEAN]]=all_data[[DATA_CLEAN]].replace('',np.nan)
    all_data[[DATA_CLEAN]]=all_data[[DATA_CLEAN]].dropna(axis=0,how='any').reset_index(drop=True)
    
    vector=Vectorizer()
    vector.tfidf(df_all_clean=all_data[DATA_CLEAN],tf_path=f'D:/分類model/大分類模型{version}/tfidf_feature_all.pkl',df_word=df_word,analyzer="word", max_features=100000, ngram_range=(1, 5))#產生tfidf向量
    
    '''
    
    tf_path = f'D:/分類model/大分類模型{version}/tfidf_feature_all.pkl'
    tf_load = pickle.load(open(tf_path, 'rb'))       
    model_selected='RFECV'
    df_pred = df.copy()  
   
    try:               
        for CATEGORY_LABEL in ['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸','乳品','蛋品','肉類','海鮮','水果','蔬菜','冰品','米油雜糧調味料','泡麵罐頭調理','巧克力','酒類','棉、紙製品','母嬰用品','居家生活','服飾鞋包','保養美妝','3C家電','保健生機','盥洗用品','禮盒伴手禮','醫療護理','18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']:
            model_path = f'D:/分類model/大分類模型{version}/model/{CATEGORY_LABEL}/model_select/{model_selected}_Classfier'
            
            df_pred=df_pred.replace(np.nan,'')
            input_fea =tf_load.transform(df_pred['data_clean'])  #transform
            
            label_model = joblib.load(model_path)
            predict_output = label_model.predict(input_fea)
            print(predict_output)
            #predict_prob_y = label_model.predict_proba(input_fea)
        
            df_pred[CATEGORY_LABEL] =predict_output
            #df_pred['機率']=[str(pre_prob[1]) for pre_prob in predict_prob_y]
            df_pred=pd.concat([df1,df_pred],axis=1)
            
            
        
    except:
        pass
    df_pred.to_excel(f"../model_test/{filename}_predict{version}.xlsx", index=False)
    
    #發票RPF
    # PRF=pd.DataFrame([])
    # df_label=pd.read_excel('../model_test/團購分類測試data_clean_label.xlsx')
    # df_pre=pd.read_excel('../model_test/團購分類測試data_clean_predict(合併).xlsx')
    # for  c in  ['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸','乳品','蛋品','肉類','海鮮','水果','蔬菜','冰品','米油雜糧調味料','泡麵罐頭調理','巧克力','酒類','棉、紙製品','母嬰用品','居家生活','服飾鞋包','保養美妝','3C家電','保健生機','盥洗用品','禮盒伴手禮','醫療護理','18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']:
    #     myMetic = Metric(np.array(df_label[[f'{c}']]), np.array(df_pre[[c]]))
    #     print(myMetic.multilabel_PRFscore(type='macro'))
    #     print(myMetic.multilabel_PRFscore(type='binary'))
    #     df2=pd.DataFrame({
    #                         'category':f'{c}',
    #                         'precision':[myMetic.multilabel_PRFscore(type='binary')[0]],
    #                         'recall':[myMetic.multilabel_PRFscore(type='binary')[1]],
    #                         'f1':[myMetic.multilabel_PRFscore(type='binary')[2]],
    #                             })
    #     PRF=pd.concat([PRF,df2],axis =0)
    # PRF.to_excel('../model_test/發票PRF.xlsx', index=False)  