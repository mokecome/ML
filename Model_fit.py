# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:43:25 2023

@author: mokecome
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.multioutput import MultiOutputClassifier
from xgboost.sklearn import XGBClassifier
from collections import defaultdict
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import cross_val_score
from config import SAMPLE_NUM,SCROING,DATA_CLEAN,LABEL
from Vector import Vectorizer
from utils.metric import Metric
import pandas as pd
import json
import joblib
import glob
import os
import shutil



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class ModelPipe:
    def __init__(self,train_x,train_y,test_x,test_y,LABEL_NAME,model_path='model/model_select/{}_Classfier',log_path=r"model/log/para.log"):
        self.LABEL_NAME=LABEL_NAME
        self.train_x=train_x
        self.train_y=train_y
        self.test_x=test_x
        self.test_y=test_y
        vector=Vectorizer()
        self.training_feature,self.train_label_y=vector.get_vector(df=pd.concat([train_x,train_y],axis=1),method='tfidf',tf_path='utils/tfidf_feature_all.pkl')
        self.testing_feature,self.test_label_y=vector.get_vector(df=pd.concat([test_x,test_y],axis=1),method='tfidf',tf_path='utils/tfidf_feature_all.pkl')
        self.model_path = model_path
        self.log_path=log_path
        self.model_dict = {
            #'R_F': [RandomForestClassifier(random_state = 1, criterion = 'entropy',n_estimators=3,min_samples_leaf = 1,max_depth=150),{'n_estimators':hp.choice('n_estimators',range (2,5,1))}],
            'RFECV': [RFECV(estimator=RandomForestClassifier(random_state=123),step=0.5, min_features_to_select=1000),{}],
            #'XGB':[XGBClassifier(eval_metric='mlogloss', seed=1),{'max_depth':hp.choice('max_depth',range (7,20,1)),'gamma':hp.quniform('gamma', 0.0, 30, 0.5),'n_estimators': hp.choice('n_estimators',np.arange(3,10,1, dtype=int))}]
        }
        self.model_selected=''
    def hyper_rf(self,params):
        clf = RandomForestClassifier(**params)
        score = cross_val_score(clf,self.training_feature,self.train_label_y,scoring=SCROING,cv=SAMPLE_NUM).mean()
        return {"loss": -score, "status": STATUS_OK} 
    def hyper_XGB(self,params):
        clf = XGBClassifier(**params)
        score = cross_val_score(clf,self.training_feature,self.train_label_y,scoring=SCROING,cv=SAMPLE_NUM).mean()
        return {"loss": -score, "status": STATUS_OK}  
    def pipe_fit(self):
        predict_dict = defaultdict(dict)                 # 用來儲存每個模型的各項metric
        for model_name, model in self.model_dict.items():# 取model_dict中的model分別訓練
            origin={}
            
            if model_name=='R_F':  
                origin=model[0].get_params(deep=True)
                best_hyperparams = fmin(self.hyper_rf,model[1], algo=tpe.suggest, max_evals=5, trials=Trials())
                origin.update(best_hyperparams)
                print(origin)
                predict_dict[model_name]['params']=origin
            elif model_name=='XGB':
                origin=model[0].get_params(deep=True)
                best_hyperparams = fmin(self.hyper_XGB,model[1], algo=tpe.suggest, max_evals=5, trials=Trials())
                origin.update(best_hyperparams)
                print(origin)
                predict_dict[model_name]['params']=origin
            elif model_name=='RFECV':
                  origin=model[0].get_params(deep=True)
                  print(origin)
                 
                  
            if self.train_y[[CATEGORY_LABEL]].shape[1]>1:#訓練多標籤
                clf=MultiOutputClassifier(model[0].set_params(**origin)).fit(self.training_feature,self.train_y[[CATEGORY_LABEL]]) 
                predict_y = clf.predict(self.testing_feature)
                myMetic = Metric(np.array(self.test_y[[CATEGORY_LABEL]]), np.array(predict_y))
                try:
                    #AUROC = myMetic.auROC() 
                    #predict_dict[model_name]["auc"] = AUROC[1]
                    predict_dict[model_name]['PRFscore_avg']=myMetic.multilabel_PRFscore(type='samples')
                except:
                    pass
                print('PRF_samples:',myMetic.multilabel_PRFscore(type='samples'))
            else:#訓練單標籤
                print('--------------',model[0])
           
                clf=model[0].set_params(**origin).fit(self.training_feature,np.ravel(self.train_y[[LABEL]])) 
                predict_y = clf.predict(self.testing_feature)
                predict_prob_y = clf.predict_proba(self.testing_feature)
                #保存test結果和概率
                myMetic = Metric(np.array(self.test_y[[LABEL]]), np.array(predict_y))
                try:
                    predict_dict[model_name]['PRFscore_avg']=myMetic.multilabel_PRFscore(type='macro')
                    predict_dict[model_name]['PRFscore_avg_binary']=myMetic.multilabel_PRFscore(type='binary')
                    print(myMetic.multilabel_PRFscore(type='macro'))
                    print(myMetic.multilabel_PRFscore(type='binary'))
                    
                except:
                    pass
              
                df1=pd.concat([self.test_x,pd.DataFrame(self.test_y),pd.DataFrame(predict_y,columns = ['pre'])],axis=1)
                df1['pre_prob']=[str(pre_prob[1]) for pre_prob in predict_prob_y]
                df1.to_excel(f'ramdom_split/{self.LABEL_NAME}_predict.xlsx',index=False) 
                df1['dif']=df1['label']-df1['pre']
                df1['add']=df1['label']+df1['pre']
                
                df1[df1['dif']==1].to_excel(f'ramdom_split/{self.LABEL_NAME}_fn.xlsx',index=False) 
                df1[df1['dif']==-1].to_excel(f'ramdom_split/{self.LABEL_NAME}_fp.xlsx',index=False)
                df1[df1['add']==0].to_excel(f'ramdom_split/{self.LABEL_NAME}_tn.xlsx',index=False)
                df1[df1['add']==2].to_excel(f'ramdom_split/{self.LABEL_NAME}_tp.xlsx',index=False)
                predict_dict[model_name]['fn']=len(df1[df1['dif']==1]) 
            
            try:    
                origin.update({'estimator':str(origin['estimator'])})#保存時
            except:
                pass
            predict_dict[model_name]['params']=origin  
            joblib.dump(clf,self.model_path.format(model_name))
        print(predict_dict)   
        self.model_selected = self._model_select(predict_dict)  # 傳入predict_dict以選擇模型  
        predict_dict['model_selected']=self.model_selected
        model_files = glob.glob(self.model_path.format('*'))
        for model_file in model_files:   
            if self.model_selected not in model_file :
                try:
                    os.remove(model_file)
                except OSError as e:
                    print(f"Error:{ e.strerror}") 
        print("====保留最佳模型====")
        
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(predict_dict, f, cls=NpEncoder)
    def _model_select(self, predict_dict):
        predict_dict = dict(sorted(predict_dict.items(), key=lambda x: (x[1]["PRFscore_avg"][2],x[1]["PRFscore_avg"][1],x[1]["PRFscore_avg"][0],),reverse=True))
        print(f"選擇模型: {list(predict_dict.keys())[0]}")
        print("====依照指標重要程度排序====")
        print(f"F1 Score : {list(predict_dict.values())[0]['PRFscore_avg'][2]}")
        print(f"Recall : {list(predict_dict.values())[0]['PRFscore_avg'][1]}")
        print(f"Precision : {list(predict_dict.values())[0]['PRFscore_avg'][0]}")
        return list(predict_dict.keys())[0]  

if __name__ == '__main__':
    
    df1=pd.DataFrame([])
    # df_word=pd.read_excel('utils/dict.xlsx')['詞']
    # all_data=pd.DataFrame([])
    # for CATEGORY_LABEL in ['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸','乳品','蛋品','肉類','海鮮','水果','蔬菜','冰品','米油雜糧調味料','泡麵罐頭調理','巧克力','酒類','棉、紙製品','母嬰用品','居家生活','服飾鞋包','保養美妝','3C家電','保健生機','盥洗用品','禮盒伴手禮','醫療護理','18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']:
    #     train_x=pd.read_excel(f'ramdom_split/{CATEGORY_LABEL}_train_x.xlsx').replace("nan",'').replace(np.nan,'')
    #     test_x=pd.read_excel(f'ramdom_split/{CATEGORY_LABEL}_test_x.xlsx').replace("nan",'').replace(np.nan,'')
       
    #     all_data=pd.concat([all_data,train_x,test_x],axis=0)
    
    # all_data[[DATA_CLEAN]]=all_data[[DATA_CLEAN]].replace('',np.nan)
    # all_data[[DATA_CLEAN]]=all_data[[DATA_CLEAN]].dropna(axis=0,how='any').reset_index(drop=True)
    
    # vector=Vectorizer()
    # vector.tfidf(df_all_clean=all_data[DATA_CLEAN],tf_path="utils/tfidf_feature_all.pkl",df_word=df_word,analyzer="word", max_features=100000, ngram_range=(1, 5))#產生tfidf向量
    
    for CATEGORY_LABEL in ['18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']:    
        train_x=pd.read_excel(f'ramdom_split/{CATEGORY_LABEL}_train_x.xlsx').replace("nan",'')
        train_y=pd.read_excel(f'ramdom_split/{CATEGORY_LABEL}_train_y.xlsx')
        test_x=pd.read_excel(f'ramdom_split/{CATEGORY_LABEL}_test_x.xlsx').replace("nan",'')
        test_y=pd.read_excel(f'ramdom_split/{CATEGORY_LABEL}_test_y.xlsx')
                   
        modelpipe=ModelPipe(train_x,train_y,test_x,test_y,LABEL_NAME=CATEGORY_LABEL)
        modelpipe.pipe_fit()
        
        
        if not os.path.exists(f'../model/{CATEGORY_LABEL}'):
            os.mkdir(f'../model/{CATEGORY_LABEL}')
            
        
       
        src = 'model/log/para.log'
        dst = f'../model/{CATEGORY_LABEL}/para.log'
        shutil.copyfile(src, dst)
         
        src = 'model/model_select'
        dst = f'../model/{CATEGORY_LABEL}/model_select'
        try:
            shutil.copytree(src, dst)
        except:
            pass 
        for u in ['dict.xlsx','stopwords.xlsx','alllist.txt','artificial_word_all.xlsx']:
            src = f'utils/{u}'
            if u=='artificial_word_all.xlsx':
                src = f'../大類別mapping/utils/{u}'
            
            if not os.path.exists(f'../model/{CATEGORY_LABEL}/utils'):
                os.mkdir(f'../model/{CATEGORY_LABEL}/utils')
            dst = f'../model/{CATEGORY_LABEL}/utils/{u}'
            shutil.copyfile(src, dst)
        
        for dif in ['fn','fp','tn','tp','predict']:
            src = f'ramdom_split/{CATEGORY_LABEL}_{dif}.xlsx'
            dst = f'../model/{CATEGORY_LABEL}/{CATEGORY_LABEL}_{dif}.xlsx'
            shutil.move(src, dst)
        
        print(CATEGORY_LABEL)
        df_v=pd.read_excel('utils/線下商品資料_V1.xlsx')[['商品代號']].astype(str)  
        for z in ['train','test']:
            df_x=pd.read_excel(f'ramdom_split/{CATEGORY_LABEL}_{z}_x.xlsx')
            df_y=pd.read_excel(f'ramdom_split/{CATEGORY_LABEL}_{z}_y.xlsx')
            df_xy=pd.concat([df_x,df_y],axis=1).astype(str)
            
            df_xy=df_xy.replace(np.nan,'')
            df_xy.to_excel(f'../model/{CATEGORY_LABEL}/{CATEGORY_LABEL}_{z}_xy.xlsx',index=False)
        
       
    
        df_fn=pd.read_excel(f'../model/{CATEGORY_LABEL}/{CATEGORY_LABEL}_fn.xlsx')         
        df_fp=pd.read_excel(f'../model/{CATEGORY_LABEL}/{CATEGORY_LABEL}_fp.xlsx')     
        df_test=pd.read_excel(f'../model/{CATEGORY_LABEL}/{CATEGORY_LABEL}_predict.xlsx')    
        myMetic = Metric(np.array(df_test[LABEL]), np.array(df_test['pre']))
        #print(myMetic.multilabel_PRFscore(type='macro'))
        print(myMetic.multilabel_PRFscore(type='binary'))  
        
        
        df_train_num=pd.read_excel(f'../model/{CATEGORY_LABEL}/{CATEGORY_LABEL}_train_xy.xlsx') 
        df_test_num=pd.read_excel(f'../model/{CATEGORY_LABEL}/{CATEGORY_LABEL}_test_xy.xlsx')
        df_pred=df_test.copy()
       
        df_pred['dif']=df_pred['label']-df_pred['pre']
        df_pred['add']=df_pred['label']+df_pred['pre']
        
        try:
            df2=pd.DataFrame({
                                'label':[f'{CATEGORY_LABEL}'],
                                'fn':[len(df_fn)],
                                'fp':[len(df_fp)],
                                'tn':[len(df_pred[df_pred['add']==0])],
                                'tp':[len(df_pred[df_pred['add']==2])],
                                'precision':[myMetic.multilabel_PRFscore(type='binary')[0]],
                                'recall':[myMetic.multilabel_PRFscore(type='binary')[1]],
                                'f1':[myMetic.multilabel_PRFscore(type='binary')[2]],
                                '訓練數量':[len(df_train_num)],
                                '測試數量':[len(df_test_num)]
                                    })
        except:
            Precision=len(df_pred[df_pred['add']==2])/(0.00001+len(df_pred[df_pred['add']==2])+len(df_pred[df_pred['dif']==-1]))
            Recall=len(df_pred[df_pred['add']==2])/(0.00001+len(df_pred[df_pred['add']==2])+len(df_pred[df_pred['dif']==1]))
            df2=pd.DataFrame({
                                'label':[f'{CATEGORY_LABEL}'],
                                'fn':[len(df_fn)],
                                'fp':[len(df_fp)],
                                'tn':[len(df_pred[df_pred['add']==0])],
                                'tp':[len(df_pred[df_pred['add']==2])],
                                'precision':[Precision],
                                'recall':[Recall],
                                'f1':[2 * Precision * Recall / (Precision + Recall)],
                                '訓練數量':[len(df_train_num)],
                                '測試數量':[len(df_test_num)]
                                    })
        df1=pd.concat([df1,df2],axis =0)
                
    df1.reset_index(inplace=True)
    src = 'utils/tfidf_feature_all.pkl'
    dst = '../model/tfidf_feature_all.pkl'
    shutil.copyfile(src, dst)


    df1.to_excel('../model/大分類模型PRF.xlsx',index=False)     


    




    

