# -*- coding:utf-8 -*- 
""" 
Created on Thu Mar 30 10:26:34 2023 
 
@author:mokecome 
""" 
#訓練標籤欄位
LABEL='label'
#清理好的欄位
DATA_CLEAN='data_clean'



#TOKEN
TOKEN='jieba'


#超參數調校  cv,scoring
TEST_SIZE=0.2
SAMPLE_NUM=5
SCROING='accuracy'

#tf-idf
ANALYZER='word'
MAX_FEATURES=100000
NGRAM_RANGE=(1,3)

#多標籤 
NOTION_LABELS=['台式','日式','中式','港式','印度','東南亞','泰式','越南','印尼','新馬','韓式','美式','法式','義式','歐式','中東'] 
NOTION_DATA_COLUMNS=['商品_clean']   #指定欄位
 
 
 
"""屬性標籤名""" 
ATTRIBUTE_LABELS={'食物材料':['ALMOND','BEEF','BERRY','CHEESE','CHICKEN','CHOCO','CORN','EGG','FRUIT','GRAPE','HONEY','LAMB','LEMON','LOBSTER','MANGO','MILK','MINT','MUSHROOM','NOODLE','NUT','PEANUT','PIG','POTATO','RADISH','REDBEAN','RICE','SALMON','SEAFOOD','TARO','TUNA','VEGETABLE','YELBEAN'],
                  '功能':['COSMETIC','NOSUGAR','SKINCARE','VEGETAR'],
                  '風味':['APPLE','BLACKBEAN','CACID','COCONUT','COFFEE','CURRY','ORANGE','PEACH','TEA','VANILLA'],
                  '場景':['BREAKFAST','LUNNER','RAINY']
                }
ATTRIBUTE_COLUMNS='' 
 
 
"""分類標籤名""" 
#多標籤
CATEGORY_LABELS=['18禁','3C家電','乳品','保健生機','保養美妝','公仔周邊玩具文具遊戲','冰品','咖啡飲品','寵物專區','居家生活','巧克力','書籍及雜誌期刊','服務、票券、點卡','服飾鞋包','棉、紙製品','母嬰用品','水果','泡麵罐頭調理','海鮮','烘焙甜點','盥洗用品','禮盒伴手禮','米油雜糧調味料','糖果零食','肉類','蔬菜','蛋品','調理食物','酒類','醫療護理','飲料','香菸'] 
CATEGORY_LABEL='飲料'                  
CATEGORY_DATA_COLUMNS=['商品名稱']


NEGATIVE_SOURCE='utils/negative_sample.xlsx'



#


