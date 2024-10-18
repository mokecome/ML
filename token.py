import jieba 
import jieba.posseg as pseg 
import pandas as pd
import numpy as np
from utils.text_process import proccessing,remove_punctuations,strQ2B
import os
import re
import shutil
import math
import time  # 内置模块, 时间
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

english_keyword_list=[i for i in  dict_list if check_english(i)]# 可以先提出來
 # with open('utils/alllist.txt','w',encoding='utf-8')as f:
 #       for i in dict_list: 
 #         f.write(str(i)+' {} {}'.format(100,'oo')+'\n')
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
       #jieba.re_han_default=re.compile('(.+)',re.U)
       new_list=[]
       cut_list=[]
       if any(_ in all_str for _ in english_keyword_list): #有該關鍵字
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
                   '''
                   else:
                        if w not in self.stopword_list and (not check_english(w)) and (p.startswith('N') or  p.startswith('A') or  p.startswith('oo')):#不是停用詞  #名詞  形容詞
                           if len(w)>1:
                               if w not in new_list:
                                  new_list.append(w)
                   '''
           new_list=[i for i in  new_list if check_english(i)]#只保留含有英文
          
           return ' '.join(new_list)
       else:
           return ''


def cut_df(file_name, n):
    df = pd.read_excel(file_name)
    df_num = len(df)
    every_epoch_num = math.floor((df_num/n))
    for index in range(n):
        file_name = f'./{index}.xlsx'
        if index < n-1:
            df_tem = df[every_epoch_num * index: every_epoch_num * (index + 1)]
        else:
            df_tem = df[every_epoch_num * index:]
        df_tem.to_excel(file_name, index=False)
  

if __name__ == '__main__': 
    #cut_df('all_spider_family.xlsx',4) 
    '''
    dict_list=pd.read_excel('dict.xlsx')['詞'].tolist()
    def pl(all_str):
        cut_list=[]
        for i in  str(all_str).strip().split(' '):
            if len(i)>1 or (len(i)==1 and i in dict_list):
                cut_list.append(i) 
        return ' '.join(cut_list)
    df['中文']=df['中文'].apply(pl)
    df['data_clean']=df['data_clean'].apply(pl)
    
    df.to_excel('123.xlsx',index=False)
    '''
   
    
    #正負樣本還原
    # for CATEGORY_LABEL in ['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸']:
    #     df_train_xy=pd.read_excel(f'../model/{CATEGORY_LABEL}/{CATEGORY_LABEL}_train_xy.xlsx')
    #     df_test_xy=pd.read_excel(f'../model/{CATEGORY_LABEL}/{CATEGORY_LABEL}_test_xy.xlsx')
    #     df_xy=pd.concat([df_train_xy,df_test_xy],axis=0)
    #     df_xy=df_xy[df_xy['電商']!='family']
    #     df_xy.rename({'商品名稱':'Detail_name'},axis='columns',inplace=True)
    #     if not os.path.exists(f'../大類別mapping/{CATEGORY_LABEL}/{CATEGORY_LABEL}_正樣本.xlsx'):
    #         df_xy[df_xy['label']==1][['電商','tag1','tag2','Detail_id','Detail_name','tag3','tag4']].to_excel(f'../大類別mapping/{CATEGORY_LABEL}/utils/{CATEGORY_LABEL}_正樣本.xlsx',index=False)
    #     if not os.path.exists(f'../大類別mapping/{CATEGORY_LABEL}/{CATEGORY_LABEL}_負樣本.xlsx'):
    #         df_xy[df_xy['label']==0][['電商','tag1','tag2','Detail_id','Detail_name','tag3','tag4']].to_excel(f'../大類別mapping/{CATEGORY_LABEL}/utils/{CATEGORY_LABEL}_負樣本.xlsx',index=False)
    

    # for CATEGORY_LABEL in ['18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區']:#
    #     df_train=pd.read_excel(f'{CATEGORY_LABEL}_train_x.xlsx')
    #     df_train[DATA_CLEAN]= df_train['商品名稱'].apply(proccessing)                    
    #     df_train[DATA_CLEAN]= df_train[DATA_CLEAN].apply(remove_punctuations)
    #     df_train[DATA_CLEAN]=df_train[DATA_CLEAN].apply(tokenizer)
    #     df_train=df_train.replace(np.nan,'')
    #     df_train.to_excel(f'{CATEGORY_LABEL}_train_x.xlsx',index=False)
        
    #     print('test')
        
    #     df_test=pd.read_excel(f'{CATEGORY_LABEL}_test_x.xlsx')
    #     df_test[DATA_CLEAN]=df_test['商品名稱'].apply(proccessing)                    
    #     df_test[DATA_CLEAN]=df_test[DATA_CLEAN].apply(remove_punctuations)  
    #     df_test[DATA_CLEAN]=df_test[DATA_CLEAN].apply(tokenizer)
    #     df_test=df_test.replace(np.nan,'')
    #     df_test.to_excel(f'{CATEGORY_LABEL}_test_x.xlsx',index=False)
        
    #---------部分分詞--------------------------------------------
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
        
    #全家
    
    # df_ps=pd.read_excel('C:/Users/mokecome/Desktop/class_label/all_spider_family(false_has1).xlsx')[['商品名稱','中文','英文','data_clean']]
    # for c in ['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸','乳品','蛋品','肉類','海鮮','水果','蔬菜','冰品','米油雜糧調味料','泡麵罐頭調理','巧克力','酒類','棉、紙製品','母嬰用品','居家生活','服飾鞋包','保養美妝','3C家電','保健生機','盥洗用品','禮盒伴手禮','醫療護理','18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']:
       
    #     df_train_test=pd.read_excel(f'C:/Users/mokecome/Desktop/label_data_20230831/{c}/{c}_train_test_xy.xlsx')[['電商','Detail_id','tag1','tag2','tag3','tag4','品番名稱','群番代號','群番名稱','商品名稱',f'{c}']].replace('nan','').replace(np.nan,'')
    #     df_ps_token=pd.merge(df_train_test,df_ps,how='inner',on=['商品名稱'])
    #     df_ps_token=df_ps_token[df_ps_token['電商']!='']
    #     df_family=df_ps_token[df_ps_token['電商']=='family']
    #     df_family[df_family[f'{c}']==1].to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{c}/{c}.xlsx',index=False)
    #     df_family[df_family[f'{c}']==0].to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{c}/{c}_n.xlsx',index=False)
        
        
    #     df_spider=df_ps_token[df_ps_token['電商']!='family']
    #     df_spider[df_spider[f'{c}']==1].to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{c}/utils/{c}_正樣本.xlsx',index=False)
    #     df_spider[df_spider[f'{c}']==0].to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{c}/utils/{c}_負樣本.xlsx',index=False)
        # df_set_d['商品名稱_temp']=df_set_d['商品名稱'].apply(proccessing) 
        # df_set_d['中文(false)']=df_set_d['商品名稱_temp'].apply(tokenizer,args=(-1,))
        # df_set_d['英文']=df_set_d['商品名稱_temp'].apply(tokenizer,args=(1,))
        # df_set_d['中文']=df_set_d['商品名稱_temp'].apply(tokenizer,args=(0,))
        # df_set_d['data_clean']= df_set_d['中文']+' '+df_set_d['英文']
        
        # df_set_d=df_set_d.drop(columns=['商品名稱_temp'])    
        # df_set_d.to_excel(f'../label_data_20230831/{c}/{c}_train_test_xy_set_d.xlsx',index=False)
        
        
       
    
    
        
        # df_ps_token=pd.read_excel(f'C:/Users/mokecome/Desktop/label_data_20230831/{c}/{c}_train_test_xy_token.xlsx')
        # df_ps_token_s=df_ps_token[df_ps_token['電商']!='family']    
        # df_ps_token_s[df_ps_token_s[f'{c}']==1].to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{c}/utils/{c}_正樣本.xlsx',index=False)
        # df_ps_token_s[df_ps_token_s[f'{c}']==0].to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{c}/utils/{c}_負樣本.xlsx',index=False)
        
        # pd.concat([df_set_d[df_set_d[f'{c}']==1],df_ps_token_s[df_ps_token_s[f'{c}']==1]],axis=0)
        # pd.concat([df_set_d[df_set_d[f'{c}']==0],df_ps_token_s[df_ps_token_s[f'{c}']==0]],axis=0)
        
    #spider_family=pd.read_excel('all_spider_family.xlsx').replace('nan','').replace(np.nan,'')
    
    family=pd.read_excel('線下商品資料_V1.xlsx')[:20000].replace('nan','').replace(np.nan,'')  #19:50 
    family['商品名稱_temp']=family['商品名稱'].apply(proccessing)  
    #family['中文(false)']=family['商品名稱_temp'].apply(tokenizer,args=(-1,))               
    family['中文']=family['商品名稱_temp'].apply(tokenizer,args=(0,))
    time_1 = time.time()
    jieba.re_han_default=re.compile('(.+)',re.U)
    family['英文']=family['商品名稱_temp'].apply(tokenizer,args=(1,))
    time_2 = time.time()    
    use_time = int(time_2) - int(time_1)
    print(f'英文总计耗时{use_time}秒')
    #family['data_clean(false)']= family['中文(false)']+' '+family['英文']
    family['data_clean']= family['中文']+' '+family['英文']
    
    family=family.drop(columns=['商品名稱_temp'])    
    family=family.replace(np.nan,'')
    family.to_excel('線下商品資料_V1_test.xlsx',index=False)
    
    
    
    # family_negative=pd.read_excel('utils/線下商品資料_V1.xlsx').replace('nan','').replace(np.nan,'')  
    # set_d_df = pd.concat([family_negative, family_negative_, family_negative_]).drop_duplicates(keep=False) #df2-df1
    # set_d_df['商品名稱_temp']=set_d_df['商品名稱'].apply(proccessing)
    # set_d_df['中文']=set_d_df['商品名稱_temp'].apply(tokenizer,args=(0,))
    # set_d_df['英文']=set_d_df['商品名稱_temp'].apply(tokenizer,args=(1,))
    # set_d_df['data_clean']= set_d_df['中文']+' '+set_d_df['英文'] 
    # set_d_df=set_d_df.drop(columns=['商品名稱_temp'])  
    # set_d_df.to_excel('utils/set_df.xlsx',index=False)
    
    
    # pd.concat([family_negative_,set_d_df],axis=0).replace(np.nan,'').to_excel('utils/family_negative_sample.xlsx',index=False) #df1+set_d_df

    
    # words=['燒酒螺']
    # for word in words:
    #     word=strQ2B(word).lower()
        # family_negative_=pd.read_excel('utils/_family_negative_sample.xlsx').replace('nan','').replace(np.nan,'')   
        # family_negative_['data_clean']=family_negative_['商品名稱'].apply(proccessing)
        # for idx,row in family_negative_.iterrows():
        #     if word in  row['data_clean']:
        #       family_negative_.loc[idx,'中文']=tokenizer(row['data_clean'],0)
        #       family_negative_.loc[idx,'英文']=tokenizer(row['data_clean'],1)
        #       family_negative_.loc[idx,'data_clean']=family_negative_.loc[idx,'中文']+' '+family_negative_.loc[idx,'英文']
              
        # family_negative=pd.read_excel('utils/family_negative_sample.xlsx').replace('nan','').replace(np.nan,'')  
        # family_negative['data_clean']=family_negative['商品名稱'].apply(proccessing)
        # for idx,row in family_negative.iterrows():
        #     if word in  row['data_clean']:
        #       family_negative.loc[idx,'中文']=tokenizer(row['data_clean'],0)
        #       family_negative.loc[idx,'英文']=tokenizer(row['data_clean'],1)
        #       family_negative.loc[idx,'data_clean']=family_negative.loc[idx,'中文']+' '+family_negative.loc[idx,'英文']
        #       print(family_negative['data_clean'][idx])
        # family_negative_.to_excel('utils/_family_negative_sample.xlsx',index=False)
        # family_negative.to_excel('utils/family_negative_sample.xlsx',index=False)
        
        # for CATEGORY_LABEL in ['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸','乳品','蛋品','肉類','海鮮','水果','蔬菜','冰品','米油雜糧調味料','泡麵罐頭調理','巧克力','酒類','棉、紙製品','母嬰用品','居家生活','服飾鞋包','保養美妝','3C家電','保健生機','盥洗用品','禮盒伴手禮','醫療護理','18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']:
        #     print(CATEGORY_LABEL,word)    
        #     df_f=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/{CATEGORY_LABEL}.xlsx').astype(str)
        #     df_f['data_clean']=df_f['商品名稱'].apply(proccessing)
        #     for idx,row in df_f.iterrows():
        #         if word in  row['data_clean']:
        #           df_f.loc[idx,'中文']=tokenizer(row['data_clean'],0)
                  
        #           df_f.loc[idx,'data_clean']=df_f.loc[idx,'中文']+' '+df_f.loc[idx,'英文']
                 
        #     df_p=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/utils/{CATEGORY_LABEL}_正樣本.xlsx').astype(str)
        #     df_p['data_clean']=df_p['商品名稱'].apply(proccessing)
        #     for idx,row in df_p.iterrows():
        #         if word in  row['data_clean']:
        #           df_p.loc[idx,'中文']=tokenizer(row['data_clean'],0)
                
        #           df_p.loc[idx,'data_clean']=df_p.loc[idx,'中文']+' '+df_p.loc[idx,'英文']
             
        #     df_n=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/utils/{CATEGORY_LABEL}_負樣本.xlsx').astype(str)
        #     df_n['data_clean']=df_n['商品名稱'].apply(proccessing)
        #     for idx,row in df_n.iterrows():
        #         if word in  row['data_clean']:
        #           df_n.loc[idx,'中文']=tokenizer(row['data_clean'],0)
                
        #           df_n.loc[idx,'data_clean']=df_n.loc[idx,'中文']+' '+df_n.loc[idx,'英文']
               
        #     df_extend=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/utils/電商擴充樣本/{CATEGORY_LABEL}_擴充電商正樣本.xlsx').astype(str)
        #     df_extend['data_clean']=df_extend['商品名稱'].apply(proccessing)
        #     for idx,row in df_extend.iterrows():
        #         if word in  row['data_clean']:
        #           df_extend.loc[idx,'中文']=tokenizer(row['data_clean'],0)
        #           df_extend.loc[idx,'data_clean']=df_extend.loc[idx,'中文']+' '+df_extend.loc[idx,'英文']
        # #電商
        #     df_f.to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/{CATEGORY_LABEL}.xlsx',index=False)
        #     df_p.to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/utils/{CATEGORY_LABEL}_正樣本.xlsx',index=False)
        #     df_n.to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/utils/{CATEGORY_LABEL}_負樣本.xlsx',index=False)
        #     df_extend.to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/utils/電商擴充樣本/{CATEGORY_LABEL}_擴充電商正樣本.xlsx',index=False)
   
    
    
    
   
   
    
    # for CATEGORY_LABEL in ['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸','乳品','蛋品','肉類','海鮮','水果','蔬菜','冰品','米油雜糧調味料','泡麵罐頭調理','巧克力','酒類','棉、紙製品','母嬰用品','居家生活','服飾鞋包','保養美妝','3C家電','保健生機','盥洗用品','禮盒伴手禮','醫療護理','18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']:
    #     df_f=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/{CATEGORY_LABEL}.xlsx').astype(str).replace('nan','').replace(np.nan,'') 
    #     df_p=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/utils/{CATEGORY_LABEL}_正樣本.xlsx').astype(str).replace('nan','').replace(np.nan,'') 
    #     df_n=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/utils/{CATEGORY_LABEL}_負樣本.xlsx').astype(str).replace('nan','').replace(np.nan,'') 
    #     df_extend=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/utils/電商擴充樣本/{CATEGORY_LABEL}_擴充電商正樣本.xlsx').astype(str).replace('nan','').replace(np.nan,'') 
    #     for s in ['']:
    #         df_f['中文']=df_f['中文'].str.replace(s,'').str.strip()
    #         df_p['中文']=df_p['中文'].str.replace(s,'').str.strip()
    #         df_n['中文']=df_n['中文'].str.replace(s,'').str.strip()
    #         df_extend['中文']=df_extend['中文'].str.replace(s,'').str.strip()
            
    #         df_f['英文']=df_f['英文'].str.lower()
    #         df_p['英文']=df_p['英文'].str.lower()
    #         df_n['英文']=df_n['英文'].str.lower()
    #         df_extend['英文']=df_extend['英文'].str.lower()
            
            
    #         df_f['data_clean']=df_f['中文']+' '+df_f['英文'].str.lower()
    #         df_p['data_clean']=df_p['中文']+' '+df_p['英文'].str.lower()
    #         df_n['data_clean']=df_n['中文']+' '+df_n['英文'].str.lower()
    #         df_extend['data_clean']=df_extend['中文']+' '+df_extend['英文'].str.lower()
  
             
    #     df_f.to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/{CATEGORY_LABEL}.xlsx',index=False)
    #     df_p.to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/utils/{CATEGORY_LABEL}_正樣本.xlsx',index=False)
    #     df_n.to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/utils/{CATEGORY_LABEL}_負樣本.xlsx',index=False)
    #     df_extend.to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{CATEGORY_LABEL}/utils/電商擴充樣本/{CATEGORY_LABEL}_擴充電商正樣本.xlsx',index=False) 
        
   
        

        