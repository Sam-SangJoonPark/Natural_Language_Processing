#!/usr/bin/env python
# coding: utf-8

# # TF - IDF 직접 구현

# In[230]:


docs = ['오늘 동물원에서 원숭이와 코끼리를 봤어',
         '동물원에서 원숭이에게 바나나를 줬어 바나나를']


# In[231]:


doc_ls = []

for doc in docs :
    doc_ls.append(doc.split())
doc_ls


# In[232]:


from collections import defaultdict
word2id = defaultdict(lambda : len(word2id))


# In[233]:


for doc in doc_ls :
    for token in doc :
        word2id[token]
word2id


# In[234]:


import numpy as np
import pandas as pd
TDM = np.zeros((len(word2id), len(doc_ls)), dtype =int)
print(TDM)

for i,doc in enumerate(doc_ls) :
    for token in doc :
        TDM[word2id[token], i] += 1    # 해당 토큰의 위치(column)
# 행렬로 표기 ( BOW와 차이점 : BOW는 1차원 배열 )
TDM


# In[235]:


TDM = pd.DataFrame(TDM, columns = doc_names)
TDM


# In[236]:


doc_names = ['문서'+str(i) for i in range(len(doc_ls))]


# In[237]:


sorted_vocab = sorted((value,key) for key, value in word2id.items())


# In[238]:


vocab = [v[1] for v in sorted_vocab]


# In[239]:


TDM['단어'] = vocab


# In[240]:


TDM = TDM.set_index('단어')


# In[241]:


TDM


# In[242]:


TDM['TF'] = 0
TDM['IDF'] = 0


# In[243]:


# TF / IDF / TF-IDF 계산


# In[244]:


TDM_0 = TDM.copy()
TDM_1 = TDM.copy()


# In[245]:


del TDM_0['문서1']
del TDM_1['문서0']


# In[246]:


TDM_0


# In[247]:


TDM_1


# In[248]:


# TF는 한 문서 내에서 등장한 단어의 총 갯수(len(docls[i])) 중에서 특정 단어의 등장 횟수입니다.

for i in range(len(TDM_0)) :
    TDM_0['TF'].iloc[i] = TDM_0['문서0'].iloc[i]/len(doc_ls[0])
    
for i in range(len(TDM_1)) :
    TDM_1['TF'].iloc[i] = TDM_1['문서1'].iloc[i]/len(doc_ls[1])


# In[249]:


TDM_0


# In[250]:


TDM_1


# In[251]:


# IDF는 그 단어가 등장한 문서의 총 갯수의 역수 입니다. 
for i in range(len(TDM_0)) :
    if TDM_0['문서0'].iloc[i] != 0 :
        TDM_0['IDF'].iloc[i] += 1
        TDM_1['IDF'].iloc[i] += 1
    if TDM_1['문서1'].iloc[i] != 0 :
        TDM_1['IDF'].iloc[i] += 1 
        TDM_0['IDF'].iloc[i] += 1
        
for i in range(len(TDM_0)) :
    TDM_0['IDF'].iloc[i] = np.log10( 2/ TDM_0['IDF'].iloc[i] )
    TDM_1['IDF'].iloc[i] = np.log10( 2/ TDM_1['IDF'].iloc[i] )


# In[252]:


# TF-IDF 계산은 TF 곱하기 IDF입니다.
TDM_1['TF-IDF'] = 0
for i in range(len(TDM_1)) :
    TDM_1['TF-IDF'].iloc[i] = TDM_1['TF'].iloc[i] * TDM_1['IDF'].iloc[i]


# In[253]:


TDM_0['TF-IDF'] = 0
for i in range(len(TDM_0)) :
    TDM_0['TF-IDF'].iloc[i] = TDM_0['TF'].iloc[i] * TDM_0['IDF'].iloc[i]


# In[254]:


TDM_1


# In[255]:


TDM_0


#  # 선생님이 짠 코드

# In[257]:


TDM = np.zeros((len(doc_ls),len(word2id)), dtype = int)
print(TDM)

#이건 좀 알아두자 행렬에서 한번에 갯수 세어주는 코드. 
for i, doc in enumerate(doc_ls) :
    for token in doc :
        TDM[i, word2id[token]] += 1 #해당 토큰의 위치 
        
TDM


# In[259]:


# 이것이 의미하는 것은??? 전체 단어수 ㅇㅋ
TDM[0].sum()


# In[261]:


# 이제 TF구하는걸 함수로 만들어보자
def computeTF(TDM) :
    doc_len = len(TDM) #문서갯수 2개!
    word_len = len(TDM[0])  # 토큰갯수는 8개임
    
    tf = np.zeros((doc_len, word_len))
    print(tf)
    # TF 계산 : 특정단어빈도 / 문서내 전체등장단어빈도
    for doc_i in range(doc_len) :
        for word_i in range(word_len) :
            tf[doc_i, word_i] = TDM[doc_i, word_i] / TDM[doc_i].sum()
    return tf
        



# In[263]:


computeTF(TDM)


# In[265]:


import math
# IDF 계산 : log(총문서수 / 단어가 등장한 문서수)

def computeIDF(TDM) :
    doc_len = len(TDM)
    word_len = len(TDM[0])
    
    idf = np.zeros(word_len)
    
    for i in range(word_len) :
        idf[i] = math.log10(doc_len / np.count_nonzero(TDM[:,i]))
    return idf


# In[266]:


computeIDF(TDM)


# In[268]:


# TF - IDF 곱
def computeTFIDF(TDM) :
    tf = computeTF(TDM)
    idf = computeIDF(TDM)
    tfidf = np.zeros(tf.shape)
    for doc_i in range(tf.shape[0]) :
        for word_i in range(tf.shape[1]) :
            tfidf[doc_i, word_i] = tf[doc_i, word_i] * idf[word_i]
            
    return tfidf
        
        


# In[269]:


computeTFIDF(TDM)


# In[271]:


import pandas as pd

sorted_vocab = sorted((value,key) for key, value in word2id.items())
print(sorted_vocab)

vocab = [v[1] for v in sorted_vocab]
print(vocab)

tfidf = computeTFIDF(TDM)
pd.DataFrame(tfidf, columns=vocab)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




