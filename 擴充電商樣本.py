#1.每次擴增 取差集
import jieba 
import jieba.posseg as pseg 
import pandas as pd
import numpy as np
import datetime
from utils.text_process import proccessing,remove_punctuations,check_english,get_maxlen_english,get_maxlen_ennum
import re
import pickle
import joblib
import os
from utils.metric import Metric
spider_dict = { 
    '烘焙甜點':[('Pxgo','tag1','烘焙甜點'),('Carrefour','tag2','烘焙食品'),('7Eleven','tag2','蛋糕 / 甜點 / 冰品')],            
    "調理食物":[('Pxgo','tag1','調理熟食'),('Carrefour','tag3','包子/饅頭/餡餅'),('Carrefour','tag3','蔥油餅/披薩 水餃/麵食'),('Carrefour','tag3','微波餐點'),('Carrefour','tag3','油炸速食'),('7Eleven','tag3','水餃 / 包子 / 港點'),('7Eleven','tag3','蔥油餅/餡餅'),('7Eleven','tag3','鍋物好料')],
    '糖果零食':[('Pxgo','tag1','糖果零食'),('Carrefour','tag3','糖果'),('Carrefour','tag3','口香糖．喉糖'),('7Eleven','tag3','餅乾'),('7Eleven','tag3','洋芋片 / 點心麵 / 脆條'),('7Eleven','tag3','糖果')],
    '飲料':[('Pxgo','tag2','太古可口可樂'),('Pxgo','tag2','水氣泡水'),('Pxgo','tag2','冷藏飲料'),('Pxgo','tag2','茶'),('Pxgo','tag2','沖泡茶品'),('Pxgo','tag2','運動汽水'),('Pxgo','tag2','果汁其他'),('Pxgo','tag2','果蔬其他'),('Pxgo','tag2','維他露'),('Pxgo','tag2','悅氏'),('Pxgo','tag2','立頓'),#味全
          ('Carrefour','tag2','水．汽水'),('Carrefour','tag3','保健養生'),('Carrefour','tag3','蔬果汁'),('Carrefour','tag3','濃縮飲料/水果醋'),('Carrefour','tag3','綠茶．烏龍茶．其他茶飲'),('Carrefour','tag3','紅茶．花茶．水果茶'),('Carrefour','tag3','奶茶'),('Carrefour','tag3','機能飲料'),('Carrefour','tag3','沖泡飲品'),('Carrefour','tag3','箱裝水'),('Carrefour','tag3','茶包'),('Carrefour','tag3','飲料沖泡'),
            ('7Eleven','tag2','茶飲料'),('7Eleven','tag2','礦泉水'),('7Eleven','tag2','茶葉 / 茶包'),('7Eleven','tag3','汽水 / 碳酸 / 運動飲料'),('7Eleven','tag3','果汁 / 醋飲'),('7Eleven','tag3','沖調飲品'),('7Eleven','tag4','各式沖泡飲 / 風味茶包'),('7Eleven','tag4','茶飲/果汁'),('7Eleven','tag4','木耳露')],
    '咖啡飲品':[('Pxgo','tag2','咖啡'),('Carrefour','tag2','咖啡'),('7Eleven','tag3','沖泡咖啡'),('7Eleven','tag3','濾掛咖啡')],
    '香菸':[('Pxgo','tag1','-'),('Carrefour','tag1','-'),('7Eleven','tag1','-')],
    '乳品':[('Pxgo','tag1','冷藏乳飲'),('Carrefour','tag2','鮮乳．調味乳'),('Carrefour','tag2','優酪乳．優格'),('Carrefour','tag2','保久乳')],
    '蛋品':[('Pxgo','tag1','蛋品')], 
    '肉類':[('Pxgo','tag2','其他肉品'),('Pxgo','tag2','牛肉類'),('Carrefour','tag2','肉品．高湯'),('7Eleven','tag3','蝦類'),('7Eleven','tag3','豬肉 / 豬肉片 / 豬排'),('7Eleven','tag3','雞肉')],
    '海鮮':[('Pxgo','tag1','海鮮類'),('Carrefour','tag2','海鮮水產'),('7Eleven','tag2','水產海鮮')], 
    '水果':[('Pxgo','tag2','水果'),('Carrefour','tag3','季節水果'),('Carrefour','tag3','福和生鮮'),('Carrefour','tag3','優果園'),('Carrefour','tag3','一起買水果'),('7Eleven','tag3','季節水果')], 
    '蔬菜':[('Pxgo','tag2','葉菜'),('Pxgo','tag2','青花芽菜'),('Pxgo','tag2','薑辛蔥'),('Carrefour','tag3','各式蔬菜'),('7Eleven','tag3','當令蔬菜 / 蔬菜箱')], 
    
    
    '泡麵罐頭調理':[('Pxgo','tag1','快煮泡麵'),('Pxgo','tag1','罐頭調味'),('Pxgo','tag2','咖哩調理包'),('Pxgo','tag2','冷藏調理'),('Carrefour','tag2','泡麵．麵條'),('Carrefour','tag3','調味品．罐頭．湯品'),('Carrefour','tag2','肉品．高湯'),('Carrefour','tag2','抹醬．蜂蜜．奶油'),('Carrefour','tag2','冷凍調理'),('7Eleven','tag1','麵食料理'),('7Eleven','tag2','水餃/ 熟食小吃')], 
    '米油雜糧調味料':[('Pxgo','tag1','米油雜糧'),('Pxgo','tag2','沖泡麥片'),('Pxgo','tag2','咖哩塊'),('Carrefour','tag2','咖哩塊'),('Carrefour','tag1','米油沖泡'),('7Eleven','tag1','麵食料理'),('7Eleven','tag3','油品 / 調味醬 / 鹽')], 
    '酒類':[('Pxgo','tag1','-'),('Carrefour','tag1','好康主題'),('7Eleven','tag1','保健生機')],
    '巧克力':[('Pxgo','tag2','巧克力'),('Carrefour','tag3','巧克力'),('Carrefour','tag3','年節伴手禮'),('7Eleven','tag3','巧克力')],
    '冰品':[('Pxgo','tag2','冰淇淋'),('Carrefour','tag2','冰品'),('Carrefour','tag2','進口好物'),('7Eleven','tag2','蛋糕 / 甜點 / 冰品'),('7Eleven','tag2','名店免排隊')],
    '棉、紙製品':[('Pxgo','tag1','紙棉製品'),('Carrefour','tag2','衛生紙'),('Carrefour','tag2','嬰童紙尿褲'),('Carrefour','tag2','女性衛生'),('Carrefour','tag3','棉花棒．繃帶'),('Carrefour','tag3','成人紙尿褲．尿墊'),('Carrefour','tag3','口罩'),('7Eleven','tag3','衛生紙/廚巾/尿布')],
    '保養美妝':[('Pxgo','tag2','臉部清潔'),('Pxgo','tag2','彩妝小物'),('Carrefour','tag2','美妝保養'),('Carrefour','tag2','專櫃美妝'),('7Eleven','tag2','臉部護理'),('7Eleven','tag2','自白肌'),('7Eleven','tag2','美容家電')],
    '居家生活':[('Pxgo','tag1','家清百貨'),('Carrefour','tag1','傢俱寢飾'),('Carrefour','tag1','日用生活'),('7Eleven','tag1','居家生活')],
    '盥洗用品':[('Pxgo','tag1','個人清潔'),('Carrefour','tag2','個人清潔'),('Carrefour','tag2','衛浴用品'),('Carrefour','tag2','專櫃美妝'),('Carrefour','tag2','美妝保養'),('7Eleven','tag1','美容保養'),('7Eleven','tag2','美容家電'),('7Eleven','tag2','家用清潔')], 
    '保健生機':[('Pxgo','tag2','保健食品'),('Pxgo','tag2','成人營養奶粉'),('Pxgo','tag2','果汁其他'),('Carrefour','tag2','營養保健食品'),('Carrefour','tag2','果汁．養生'),('7Eleven','tag1','保健生機')],
    '3C家電':[('Pxgo','tag1','生活家電'),('Pxgo','tag1','手機通訊'),('Carrefour','tag1','熱門3C'),('Carrefour','tag1','生活家電'),('7Eleven','tag1','家電')], 
    '服飾鞋包': [('Pxgo','tag1','服飾內衣'),('Pxgo','tag2','紙褲內衣'),('Carrefour','tag1','服飾鞋包'),('7Eleven','tag1','服飾鞋包')],
    '寵物專區':[('Carrefour','tag2','寵物用品')],
    '母嬰用品':[('Carrefour','tag1','嬰童保健'),('7Eleven','tag2','身體保養/清潔')],
    #禮盒伴手禮
    '醫療護理':[('Pxgo','tag2','衛生保健'),('Pxgo','tag2','身體清潔'),('Pxgo','tag2','聯合利華'),('Pxgo','tag2','康那香'),('Pxgo','tag2','嬰兒用品'),('Pxgo','tag2','口腔清潔'),('Carrefour','tag2','居家護理'),('7Eleven','tag1','保健生機'),('7Eleven','tag2','身體保養/清潔'),('7Eleven','tag2','臉部護理')],
    '公仔周邊玩具文具遊戲':[('Pxgo','tag2','文具'),('Carrefour','tag1','休閒娛樂'),('Carrefour','tag1','傢俱寢飾'),('Carrefour','tag2','電玩．遊戲機'),('7Eleven','tag2','icash2.0'),('7Eleven','tag2','嬰童用品')],
    '服務、票券、點卡':[('Pxgo','tag1','-'),('Carrefour','tag1','休閒娛樂'),('7Eleven','tag1','-')],
    '書籍及雜誌期刊':[('Pxgo','tag1','-'),('Carrefour','tag1','-'),('7Eleven','tag1','-')],
    '18禁':[('7Eleven','tag2','用品/醫材/18禁情趣商品')],
    '加工食品':[('Pxgo','tag2','肉鬆'),('Pxgo','tag2','起士奶油')
              ,('Carrefour','tag3','奶油．起司'),('Carrefour','tag3','豆類'),('Carrefour','tag3','加工製品'),('Carrefour','tag3','瓜子．花生．堅果'),('Carrefour','tag3','豆乾．肉乾．海苔'),('Carrefour','tag3','紫菜/魚乾/烏魚子'),('Carrefour','tag3','雞蛋．豆製品'),('Carrefour','tag3','各式火鍋料'),('Carrefour','tag3','菇菌類')
              ,('7Eleven','tag3','各式調味醬料/乾貨'),('7Eleven','tag3','滷味 / 涼拌')]
}
def tokenizer(all_str,flag): 
    dict_list=pd.read_excel('utils/dict.xlsx')['詞'].tolist()
    stopword_list=pd.read_excel('utils/stopwords.xlsx')['stopword'].tolist()  
    with open('utils/alllist.txt','w',encoding='utf-8')as f:
          for i in dict_list: 
            f.write(str(i)+' {} {}'.format(100,'oo')+'\n')   
    if flag==0:
        jieba.load_userdict('utils/alllist.txt') 
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
                            if len(w)>1:
                                if w not in new_list:
                                    new_list.append(w)
        print(' '.join(new_list)) 
        return ' '.join(new_list)
   
    if flag==1:
        jieba.load_userdict('utils/alllist.txt') 
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
           
            print(' '.join(new_list))#只保留英文 
            return ' '.join(new_list)
        else:
            return ''

# n_time = datetime.datetime.now()
# df_Carrefour=pd.read_csv('D:/ECommerceDataSpider/Carrefour/{}年{}月/Carrefour_detail.csv'.format(n_time.year,n_time.month))[['電商','tag1','tag2','tag3','Detail_id','Detail_name']]
# df_Pxgo=pd.read_csv('D:/ECommerceDataSpider/Pxgo/{}年{}月/Pxgo_detail.csv'.format(n_time.year,n_time.month))[['電商','tag1','tag2','Detail_id','Detail_name']]
# df_7_11=pd.read_csv('D:/ECommerceDataSpider/7Eleven/{}年{}月/7Eleven_detail.csv'.format(n_time.year,n_time.month))[['電商','tag1','tag2','tag3','tag4','Detail_id','Detail_name']]
# print('特徵字過濾資料')
# for category in ['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸','乳品','蛋品','肉類','海鮮','水果','蔬菜','冰品','米油雜糧調味料','泡麵罐頭調理','巧克力','酒類','棉、紙製品','母嬰用品','居家生活','服飾鞋包','保養美妝','3C家電','保健生機','盥洗用品','禮盒伴手禮','醫療護理','18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']:
#     if not os.path.exists(f'../大類別mapping/{category}/utils/電商擴充樣本'):
#       os.mkdir(f'../大類別mapping/{category}/utils/電商擴充樣本') 
#     try:
#         df_artificial=pd.read_excel('../大類別mapping/utils/artificial_word_all.xlsx')[category]#        
#         print('類別',category)
#         df_data_Pxgo=pd.DataFrame([])
#         df_data_Carrefour=pd.DataFrame([])
#         df_data_7_11=pd.DataFrame([])
#         for word in [i.replace('(','\(').replace(')','\)') for i in df_artificial.tolist() if i==i]:
#             df_data_Pxgo=pd.concat([df_data_Pxgo,df_Pxgo[df_Pxgo['Detail_name'].str.contains(word)]],axis=0)
#             df_data_Carrefour=pd.concat([df_data_Carrefour,df_Carrefour[df_Carrefour['Detail_name'].str.contains(word)]],axis=0)
#             df_data_7_11=pd.concat([df_data_7_11,df_7_11[df_7_11['Detail_name'].str.contains(word)]],axis=0) 
#         dfpc7=pd.concat([df_data_Pxgo,df_data_Carrefour,df_data_7_11],axis=0).rename(columns={'Detail_name':'商品名稱'})      
#         dfpc7.to_excel(f'../大類別mapping/{category}/電商.xlsx',index=False)
#     except:
#           pass
# #有tag過濾 
# for category in ['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸','乳品','蛋品','肉類','海鮮','水果','蔬菜','冰品','米油雜糧調味料','泡麵罐頭調理','巧克力','酒類','棉、紙製品','母嬰用品','居家生活','服飾鞋包','保養美妝','3C家電','保健生機','盥洗用品','禮盒伴手禮','醫療護理','18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']:
#     df=pd.read_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{category}/電商.xlsx')      
#     df_data_Pxgo=df[df['電商']=='全聯']
#     df_data_Carrefour=df[df['電商']=='carrefour']
#     df_data_7_11=df[df['電商']=='7-11']

#     Pxgo_spider=pd.DataFrame([])  
#     Carrefour_spider=pd.DataFrame([])  
#     Seven_spider=pd.DataFrame([])
#     try:
#         for s in spider_dict[category]:            
#             if s[0]=='Pxgo':
#                 Pxgo_spider=pd.concat([Pxgo_spider,df_data_Pxgo[df_data_Pxgo[s[1]]==s[2]]],axis=0)   
#             elif s[0]=='Carrefour':
#                 Carrefour_spider=pd.concat([Carrefour_spider,df_data_Carrefour[df_data_Carrefour[s[1]]==s[2]]],axis=0)
#             elif s[0]=='7Eleven':
#                 Seven_spider=pd.concat([Seven_spider,df_data_7_11[df_data_7_11[s[1]]==s[2]]],axis=0)
#         df_spider=pd.concat([Pxgo_spider,Carrefour_spider,Seven_spider],axis=0)
#         df_spider=df_spider.replace(np.nan,'')
#         df_spider.to_excel(f'C:/Users/mokecome/Desktop/大類別mapping/{category}/電商.xlsx',index=False) 
#     except:
#         pass



    
# for category in ['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸','乳品','蛋品','肉類','海鮮','水果','蔬菜','冰品','米油雜糧調味料','泡麵罐頭調理','巧克力','酒類','棉、紙製品','母嬰用品','居家生活','服飾鞋包','保養美妝','3C家電','保健生機','盥洗用品','禮盒伴手禮','醫療護理','18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']: 
#     df_origin_n=pd.read_excel(f'../大類別mapping/{category}/utils/{category}_負樣本.xlsx')[['電商','tag1','tag2','Detail_id','商品名稱','tag3','tag4']]
#     df_origin_p=pd.read_excel(f'../大類別mapping/{category}/utils/{category}_正樣本.xlsx')[['電商','tag1','tag2','Detail_id','商品名稱','tag3','tag4']]
#     df_origin=pd.concat([df_origin_p,df_origin_n],axis=0)
#     df_new=pd.read_excel(f'../大類別mapping/{category}/電商.xlsx')  
#     set_d_df = pd.concat([df_new, df_origin, df_origin]).drop_duplicates(keep=False) 

#     set_d_df.to_excel(f'../大類別mapping/{category}/utils/電商擴充樣本/{category}_電商差集.xlsx',index=False)
    
#分詞
# for category in ['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸','乳品','蛋品','肉類','海鮮','水果','蔬菜','冰品','米油雜糧調味料','泡麵罐頭調理','巧克力','酒類','棉、紙製品','母嬰用品','居家生活','服飾鞋包','保養美妝','3C家電','保健生機','盥洗用品','禮盒伴手禮','醫療護理','18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']: 
#     df=pd.read_excel(f'../大類別mapping/{category}/utils/電商擴充樣本/{category}_電商差集.xlsx') 
#     #df=df.rename({'Detail_name':'商品名稱'},axis='columns')
#     df['data_clean']=df['商品名稱'].apply(proccessing)
#     df['中文']=df['data_clean'].apply(tokenizer,args=(0,)).str.replace("nan","")
#     df['英文']=df['data_clean'].apply(tokenizer,args=(1,)).str.replace("nan","")
#     df['data_clean']=df['中文']+' '+df['英文']
#     df.to_excel(f"../大類別mapping/{category}/utils/電商擴充樣本/{category}_擴充電商正樣本.xlsx", index=False)

    
#模型預測->人工FP
tf_path = 'C:/Users/mokecome/Desktop/model/tfidf_feature_all.pkl'
tf_load = pickle.load(open(tf_path, 'rb'))
model_selected='RFECV' 
PRF=pd.DataFrame([]) 
data=pd.read_excel('../大類別mapping/線下商品資料_V1_token.xlsx')
for CATEGORY_LABEL in ['烘焙甜點','調理食物','糖果零食','飲料','咖啡飲品','香菸','乳品','蛋品','肉類','海鮮','水果','蔬菜','冰品','米油雜糧調味料','泡麵罐頭調理','巧克力','酒類','棉、紙製品','母嬰用品','居家生活','服飾鞋包','保養美妝','3C家電','保健生機','盥洗用品','禮盒伴手禮','醫療護理','18禁','公仔周邊玩具文具遊戲','服務、票券、點卡','書籍及雜誌期刊','寵物專區','加工食品']: 

    model_path = f'../model/{CATEGORY_LABEL}/model_select/{model_selected}_Classfier'      
    label_model = joblib.load(model_path)
    input_fea =tf_load.transform(data['data_clean']) 
    
    
    predict_output = label_model.predict(input_fea)
    data[CATEGORY_LABEL] =predict_output

data.to_excel('../model/線下商品資料_V1_token_predict.xlsx',index=False)

# #擴充正樣本
# df[df[f'{d}']==1].to_excel(f'../大類別mapping/{d}/utils/電商擴充樣本/{d}_預測擴充電商正樣本.xlsx', index=False)
# #擴充負樣本
# df[df[f'{d}']==0].to_excel(f'../大類別mapping/{d}/utils/電商擴充樣本/{d}_預測擴充電商負樣本.xlsx', index=False)