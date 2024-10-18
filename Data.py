# -*- coding:utf-8 -*- 
""" 
Created on Wed Mar 29 10:38:13 2023 
 
@author:mokecome 
""" 
from utils.text_process import proccessing,remove_punctuations,check_english,get_maxlen_english,get_maxlen_ennum,strQ2B
import pandas as pd 
import jieba 
import jieba.posseg as pseg 
import numpy as np 
from config import  CATEGORY_DATA_COLUMNS,DATA_CLEAN,LABEL,SAMPLE_NUM,TEST_SIZE,TOKEN,NEGATIVE_SOURCE
from sklearn.model_selection import train_test_split 
import re
import os

class DataProcess:   
    def __init__(self,data_columns,label_columns,clean=False):
        self.dict_list=pd.read_excel('utils/dict.xlsx')['詞'].tolist()
        self.english_keyword_list=[i for i in  self.dict_list if check_english(i)]
        self.stopword_list=pd.read_excel('utils/stopwords.xlsx')['stopword'].tolist()
        with open('utils/alllist.txt','w',encoding='utf-8')as f:
             for i in self.dict_list:
                f.write(str(i)+' {} {}'.format(100,'oo')+'\n')
        self.jieba=jieba
        self.jieba.load_userdict('utils/alllist.txt') 
        
        self.data_columns=data_columns 
        self.label_columns=label_columns
        self.label_column=label_columns[0]
        self.clean=clean 
    def data_check(self,df,columns_check):#一定要有的欄位 
        if  isinstance(columns_check,str):
            columns_check=[columns_check]         
        if not set(columns_check).issubset(df.columns):
            raise ValueError('缺少訓練數據{}'.format(set(columns_check)-set(df.columns)))
   
    def create_label(self,df_,Marking=False):#有標籤才要產生                
        if len(self.label_columns)>1:    
            one_hots=np.array(df_[self.label_columns]).tolist()
            df_[LABEL]=pd.DataFrame([int(''.join(map(str, one_hot)),2) for one_hot in one_hots])
        else:
            df_[LABEL]=df_[self.label_columns]
        return df_
            
    def data_clean(self,df_):
        for c in df_.columns:#讀檔 self.data_columns
            if 'data_clean' in c:
                #比對字典
                self.clean=True 
                return df_
        if not self.clean:         
            df_[DATA_CLEAN]=0
            for d in  self.data_columns:#商品名稱
                df_[DATA_CLEAN]=df_[DATA_CLEAN].astype(str)+' '+df_[d].astype(str)             
            df_[DATA_CLEAN]=df_[DATA_CLEAN].apply(proccessing)#remove_word strQ2B lower                    
            return df_
    def tokenizer(self,all_str,token,flag): 
        if token=='jieba':
            if flag==-1:#cut_all = False，保留所有1個字的斷詞結果  
               new_list=[]
               cut_list=[]
               all_str=remove_punctuations(all_str)
               cut_list=[i for i in  all_str.strip().split(' ')]        
               
               for text in cut_list:
                   text_after_jieba=self.jieba.cut(text,cut_all=False) 
                   for cut_str in text_after_jieba:#分詞後結果 
                       seq_list=pseg.cut(cut_str)
                       for w,p in seq_list:
                            if w in self.dict_list:
                               new_list.append(w) #只能是關鍵字len(w)=1
                            else:
                                 if w not in self.stopword_list and (p.startswith('N') or  p.startswith('A') or  p.startswith('oo')):#不是停用詞  #名詞  形容詞                            
                                    if w not in new_list:
                                       new_list.append(w)
                              
               return ' '.join(new_list)
           
            if flag==0:
               new_list=[]
               cut_list=[]
               all_str=remove_punctuations(all_str)
               cut_list=[i for i in  all_str.strip().split(' ')]
                   
               for text in cut_list:
                   text_after_jieba=self.jieba.cut(text,cut_all=True)
                   for cut_str in text_after_jieba:#分詞後結果 
                       seq_list=pseg.cut(cut_str)
                       for w,p in seq_list:
                            if w in self.dict_list:
                               new_list.append(w) #只能是關鍵字len(w)=1
                            else:
                                 if w not in self.stopword_list and (p.startswith('N') or  p.startswith('A') or  p.startswith('oo')):#不是停用詞  #名詞  形容詞 
                                        if w not in new_list:
                                           new_list.append(w)
               print(' '.join(new_list)) 
               return ' '.join(new_list)
           
            if flag==1:
               #self.jieba.re_han_default=re.compile('(.+)',re.U)
               new_list=[]
               cut_list=[]      
               if any(_ in all_str for _ in self.english_keyword_list): #有該關鍵字 
                   text_after_jieba=self.jieba.cut(all_str,cut_all=False)
                   for cut_str in text_after_jieba:#分詞後結果 
                       seq_list=pseg.cut(cut_str)
                       for w,p in seq_list:
                           if w in self.dict_list:
                               if len(w)<4 and w.encode('UTF-8').isalpha(): #為英文且長度小於4
                                  if get_maxlen_english(all_str)==w:
                                      new_list.append(w) 
                               elif len(w)<5 and w.encode('UTF-8').isalnum():#英數字
                                   if get_maxlen_ennum(all_str)==w:
                                       new_list.append(w) 
                                        
                               else:#中英文 中文
                                   new_list.append(w)
                           '''
                           else:
                                if w not in df_stopword and (not check_english(w)) and (p.startswith('N') or  p.startswith('A') or  p.startswith('oo')):#不是停用詞  #名詞  形容詞
                                   if len(w)>1:
                                       if w not in new_list:
                                          new_list.append(w)
                           '''
                   new_list=[i for i in  new_list if check_english(i)]#只保留含有英文
                   
                   print(' '.join(new_list))#只保留英文 
                   return ' '.join(new_list)
               else:
                   return ''
     
            
        return all_str 
     
    def df_add_sample(self,df,sample_num):  
        all_keys=df[LABEL].unique().tolist()#單標籤LABEL 
        c_df=pd.DataFrame()
        for key in all_keys:        
            if df.loc[df[LABEL]==key].shape[0]< sample_num:
                c_df=pd.concat([c_df,df.loc[df[LABEL]==key]])      
        df_add=pd.concat([c_df]*sample_num,axis=0)#擴充樣本   
        df_data=pd.concat([df,df_add],axis=0) 
        print('原數量',len(df))
        print('增加樣本數量',len(c_df)*sample_num)
        print('擴充後樣本數量',len(df_data))
        print(df_data)
        return  df_data 
    # def no_split(self,df):
    #     Folder_name='no_split'
    #     train_x=df[['電商','中文','英文']+self.data_columns+[DATA_CLEAN]] 
    #     train_y=df[self.label_columns+[LABEL]]
    #     test_x=df_test[['電商','中文','英文']+self.data_columns+[DATA_CLEAN]] 
    #     test_y=df_test[self.label_columns+[LABEL]]
        
    #     train_x.to_excel(f'{Folder_name}/{self.label_column}_train_x.xlsx',index=False)
    #     train_y.to_excel(f'{Folder_name}/{self.label_column}_train_y.xlsx',index=False)
    #     test_x.to_excel(f'{Folder_name}/{self.label_column}_test_x.xlsx',index=False)
    #     test_y.to_excel(f'{Folder_name}/{self.label_column}_test_y.xlsx',index=False)
        
    def random_train_test(self,df,test_ratio):
        Folder_name='ramdom_split'
        np.random.seed(1)
        shuffled_indices=np.random.permutation(len(df))
        test_set_size=int(len(df)*test_ratio)
        df_test=df.iloc[shuffled_indices[:test_set_size]].reset_index(drop=True)
        df_train=df.iloc[shuffled_indices[test_set_size:]].reset_index(drop=True)
        
        train_x=df_train[['電商','Detail_id','tag1','tag2','tag3','tag4','品番名稱','群番代號','群番名稱','中文','英文']+self.data_columns+[DATA_CLEAN]] 
        train_y=df_train[self.label_columns+[LABEL]]
        test_x=df_test[['電商','Detail_id','tag1','tag2','tag3','tag4','品番名稱','群番代號','群番名稱','中文','英文']+self.data_columns+[DATA_CLEAN]] 
        test_y=df_test[self.label_columns+[LABEL]]
        
        if not os.path.exists(f'{Folder_name}'):
           os.mkdir(f'{Folder_name}')
        train_x.to_excel(f'{Folder_name}/{self.label_column}_train_x.xlsx',index=False)
        train_y.to_excel(f'{Folder_name}/{self.label_column}_train_y.xlsx',index=False)
        test_x.to_excel(f'{Folder_name}/{self.label_column}_test_x.xlsx',index=False)
        test_y.to_excel(f'{Folder_name}/{self.label_column}_test_y.xlsx',index=False)
        
        return train_x,test_x,train_y,test_y
         
    def get_train_test_split(self,df,test_ratio): #分
        Folder_name='train_test_split'
        features=df[['電商','Detail_id','tag1','tag2','tag3','tag4','品番名稱','群番代號','群番名稱','中文','英文']+self.data_columns+[DATA_CLEAN]] #可加入未分詞的欄位
        target=df[self.label_columns+[LABEL]]
        try:    
            train_x,test_x,train_y,test_y=train_test_split(features,target,test_size=test_ratio,random_state=1,stratify=target)
        except Exception as err: #erro
            print(f"Unexpected {err=}, {type(err)=}")
            raise
        if not os.path.exists(f'{Folder_name}'):
           os.mkdir(f'{Folder_name}')
        train_x.to_excel(f'{Folder_name}/{self.label_column}_train_x.xlsx',index=False)
        train_y.to_excel(f'{Folder_name}/{self.label_column}_train_y.xlsx',index=False)
        test_x.to_excel(f'{Folder_name}/{self.label_column}_test_x.xlsx',index=False)
        test_y.to_excel(f'{Folder_name}/{self.label_column}_test_y.xlsx',index=False)
        return train_x,test_x,train_y,test_y
     
    def run(self):
        df_p=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{self.label_column}/utils/{self.label_column}_正樣本.xlsx')
        self.data_check(df_p,self.data_columns)#正樣本欄位檢查
        df_p=self.data_clean(df_p)
        if not self.clean:
            df_p['中文']=df_p[DATA_CLEAN].apply(self.tokenizer,args=(TOKEN,0))
            df_p['英文']=df_p[DATA_CLEAN].apply(self.tokenizer,args=(TOKEN,1)) 
            df_p[DATA_CLEAN]=df_p['中文']+' '+df_p['英文']
            df_p.to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{self.label_column}/utils/{self.label_column}_正樣本.xlsx',index=False)
        '''擴充正負樣本(關鍵詞篩選)
        if os.path.exists(f'../大類別mapping/{self.label_column}/utils/電商擴充樣本/{self.label_column}_擴充電商正樣本.xlsx'):
            machine_p=pd.read_excel(f'../大類別mapping/{self.label_column}/utils/電商擴充樣本/{self.label_column}_擴充電商正樣本.xlsx')
            df_p=pd.concat([df_p,machine_p],axis=0)  
        '''    
        df_p[self.label_column]=1
        #電商負樣本
        df_negative=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{self.label_column}/utils/{self.label_column}_負樣本.xlsx')
        df_negative=self.data_clean(df_negative)
        if not self.clean:
            df_negative['中文']=df_negative[DATA_CLEAN].apply(self.tokenizer,args=(TOKEN,0))
            df_negative['英文']=df_negative[DATA_CLEAN].apply(self.tokenizer,args=(TOKEN,1)) 
            df_negative[DATA_CLEAN]=df_negative['中文']+' '+df_negative['英文']
            df_negative.to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{self.label_column}/utils/{self.label_column}_負樣本.xlsx',index=False)
        df_negative[self.label_column]=0
        #全家正樣本
        df_family_p=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{self.label_column}/{self.label_column}.xlsx')
        df_family_p=self.data_clean(df_family_p)
        if not self.clean:
            df_family_p['中文']=df_family_p[DATA_CLEAN].apply(self.tokenizer,args=(TOKEN,0))         
            df_family_p['英文']=df_family_p[DATA_CLEAN].apply(self.tokenizer,args=(TOKEN,1)) 
            df_family_p[DATA_CLEAN]=df_family_p['中文']+' '+df_family_p['英文']
            df_family_p.to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{self.label_column}/{self.label_column}.xlsx',index=False) 
        df_family_p[self.label_column]=1
        #全家負樣本
        try:
            df_family_n=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{self.label_column}/{self.label_column}_n.xlsx')
            df_family_n=self.data_clean(df_family_n)
            if not self.clean:
                df_family_n['中文']=df_family_n[DATA_CLEAN].apply(self.tokenizer,args=(TOKEN,0))
                df_family_n['英文']=df_family_n[DATA_CLEAN].apply(self.tokenizer,args=(TOKEN,1)) 
                df_family_n[DATA_CLEAN]=df_family_n['中文']+' '+df_family_n['英文']
        except:
            df_family_n=pd.DataFrame([])
            df_family_n[self.label_column]=0
            
       
       
        # all_category=['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸','乳品','蛋品','肉類','海鮮','水果','蔬菜','冰品','米油雜糧調味料','泡麵罐頭調理','巧克力','酒類','棉、紙製品','母嬰用品','居家生活','服飾鞋包','保養美妝','3C家電','保健生機','盥洗用品','禮盒伴手禮','醫療護理','18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']
        # df_n_interaction=pd.DataFrame([])
        # df_interaction_f=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{self.label_column}/{self.label_column}.xlsx')[['電商','Detail_id','品番名稱','群番代號','群番名稱','商品名稱','data_clean','中文','英文']]
        # for c in  list(set(all_category)-set(self.label_columns)):
        #     family_interaction=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{c}/{c}.xlsx')[['電商','Detail_id','品番名稱','群番代號','群番名稱','商品名稱','data_clean','中文','英文']]
        #     intersected_df = pd.merge(df_interaction_f,family_interaction,how='inner')
        #     if len(intersected_df)==0:
        #         df_n_interaction=pd.concat([df_n_interaction,family_interaction],axis=0)
        # df_n_interaction[self.label_column]=0
        # df_n_interaction.to_excel(f'{self.label_column}_n_interaction.xlsx',index=False)
       
        
        df_spider=pd.concat([df_p,df_negative],axis=0).drop_duplicates(subset=self.data_columns,keep="first")#電商正+負樣本
        df_family=pd.concat([df_family_p,df_family_n],axis=0).drop_duplicates(subset=self.data_columns,keep="first")# ,df_n_interaction
        
        df=pd.concat([df_spider,df_family],axis=0)
        df=self.create_label(df)#設定標籤
        df=df.replace(np.nan,'').replace('nan','')
        
 
        #訓練  
        print('FOR 資料拆分')
        train_x,test_x,train_y,test_y=self.random_train_test(df,test_ratio=1/SAMPLE_NUM) 
        
        # df=self.df_add_sample(df,SAMPLE_NUM)
        # train_x,test_x,train_y,test_y=self.get_train_test_split(df,test_size=1/SAMPLE_NUM)       
        print('FOR 超參數調優')
        
        keyword_fake_data=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{self.label_column}/{self.label_column}_fake_data(keyword).xlsx')
        train_x=pd.concat([train_x,keyword_fake_data],axis=0)  
        keyword_fake_data_y=pd.DataFrame({f'{self.label_column}':len(keyword_fake_data)*[1],LABEL:len(keyword_fake_data)*[1]})
        train_y=pd.concat([train_y,keyword_fake_data_y],axis=0)  
        
        df_train_xy=self.df_add_sample(pd.concat([train_x,train_y],axis=1),SAMPLE_NUM)
        train_x=df_train_xy[['電商','Detail_id','tag1','tag2','tag3','tag4','品番名稱','群番代號','群番名稱','中文','英文']+self.data_columns+[DATA_CLEAN]]
        train_y=df_train_xy[self.label_columns+[LABEL]]
        return train_x,test_x,train_y,test_y
'''test?
class Data_Split:
    def __init__(self,df,label_columns): 
        self.df=df
        self.label_columns=label_columns
    def df_add_sample(self,df,sample_num):  
        all_keys=df[LABEL].unique().tolist()#單標籤LABEL 
        c_df=pd.DataFrame()
        for key in all_keys:        
            if df.loc[df[LABEL]==key].shape[0]< sample_num:
                c_df=pd.concat([c_df,df.loc[df[LABEL]==key]])      
        df_add=pd.concat([c_df]*sample_num,axis=0)#擴充樣本   
        df_data=pd.concat([df,df_add],axis=0) 
        print('原數量',len(df))
        print('增加樣本數量',len(c_df)*sample_num)
        print('擴充後樣本數量',len(df_data))
        print(df_data)
        return  df_data 
    
    def random_train_test(self,df,test_size=TEST_SIZE):
        np.random.seed(1)
        shuffled_indices=np.random.permutation(len(df))
        test_set_size=int(len(df)*test_size)
        df_test=df.iloc[shuffled_indices[:test_set_size]].reset_index(drop=True)
        df_train=df.iloc[shuffled_indices[test_set_size:]].reset_index(drop=True)
        
        train_x=df_train[['電商','Detail_id','tag1','tag2','tag3','tag4','品番名稱','群番代號','群番名稱','中文','英文']+self.data_columns+[DATA_CLEAN]] 
        train_y=df_train[self.label_columns+[LABEL]]
        test_x=df_test[['電商','Detail_id','tag1','tag2','tag3','tag4','品番名稱','群番代號','群番名稱','中文','英文']+self.data_columns+[DATA_CLEAN]] 
        test_y=df_test[self.label_columns+[LABEL]]
        
        return train_x,test_x,train_y,test_y    
    
    def get_train_test_split(self,test_size=TEST_SIZE): #分
        features=self.df[['電商','Detail_id','tag1','tag2','tag3','tag4','品番名稱','群番代號','群番名稱']+self.data_columns+[DATA_CLEAN]] #可加入未分詞的欄位
        target=self.df[self.label_columns+[LABEL]]
        try:    
            train_x,test_x,train_y,test_y=train_test_split(features,target,test_size=test_size,random_state=1,stratify=target)
        except Exception as err: #erro
            print(f"Unexpected {err=}, {type(err)=}")
            raise
        return train_x,test_x,train_y,test_y
    def run(self):
        print('FOR 資料拆分')
        self.df=self.df_add_sample(self.df,SAMPLE_NUM)
        train_x,test_x,train_y,test_y=self.get_train_test_split(test_size=1/SAMPLE_NUM)       
        print('FOR 超參數調優')
        df_train_xy=self.df_add_sample(pd.concat([train_x,train_y],axis=1),SAMPLE_NUM)
        train_x=df_train_xy[['電商','Detail_id','tag1','tag2','tag3','tag4','品番名稱','群番代號','群番名稱']+self.data_columns+[DATA_CLEAN]]
        train_y=df_train_xy[self.label_columns+[LABEL]]
        return train_x,test_x,train_y,test_y
'''    
if __name__=='__main__':
    #多線程
    #更新字典和停用詞
    # a=[]   
    # df=pd.read_excel('../大類別mapping/utils/artificial_word_all.xlsx')
    # for j in df.columns:
    #     a.extend(df[j].tolist())  
            
    # a=[strQ2B(aa) for aa in a if (aa==aa and aa!='')] 
    # a=[aa.lower() for aa in a if (aa==aa and aa!='')] 
    
    # df_a=pd.DataFrame({'詞':a})
    # df_dict=pd.read_excel('utils/dict.xlsx')
    # df_dict['詞']=df_dict['詞'].apply(strQ2B)
    # df_dict['詞']=df_dict['詞'].str.lower()
    # pd.concat([df_a,df_dict],axis=0).drop_duplicates().reset_index(drop=True).to_excel('utils/dict.xlsx',index=False)
    # df_stop=pd.read_excel('utils/stopwords.xlsx')
    # s=list(set(df_stop['stopword'])-set(df_dict['詞']))
    # pd.DataFrame({'stopword':s}).to_excel('utils/stopwords.xlsx',index=False)
    
    
    #分詞和產生訓練資料
    for CATEGORY_LABEL in ['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸','乳品','蛋品','肉類','海鮮','水果','蔬菜','冰品','米油雜糧調味料','泡麵罐頭調理','巧克力','酒類','棉、紙製品','母嬰用品','居家生活','服飾鞋包','保養美妝','3C家電','保健生機','盥洗用品','禮盒伴手禮','醫療護理','18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']:
        DP=DataProcess(data_columns=CATEGORY_DATA_COLUMNS,label_columns=[CATEGORY_LABEL])
        train_x,test_x,train_y,test_y=DP.run()
        
       
        train_x.to_excel(f'ramdom_split/{CATEGORY_LABEL}_train_x.xlsx',index=False)
        train_y.to_excel(f'ramdom_split/{CATEGORY_LABEL}_train_y.xlsx',index=False)
        test_x.to_excel(f'ramdom_split/{CATEGORY_LABEL}_test_x.xlsx',index=False)
        test_y.to_excel(f'ramdom_split/{CATEGORY_LABEL}_test_y.xlsx',index=False)
        
    
    
  
    
  
    
        
        
        
        
        
        
   
