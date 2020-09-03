#!/usr/bin/env python
# coding: utf-8

# In[32]:


docs = ['동물원 코끼리',
       '동물원 원숭이 바나나',
       '엄마 코끼리 아기 코끼리',
       '원숭이 바나나 코끼리 바나나']


# In[39]:


doc_ls = []
for doc in docs :
    doc_ls.append(doc.split(' '))
print(doc_ls)

    
from collections import defaultdict
word2id = defaultdict(lambda : len(word2id))

for doc in doc_ls :
    for token in doc :
        word2id[token]
word2id


# In[44]:


word2id


# In[34]:


# 이거 실습하고 설명하고 끝 # 


# In[59]:


import numpy as np

TDM = np.zeros((len(word2id), len(doc_ls)), dtype =int)
print(TDM)

for i,doc in enumerate(doc_ls) :
    for token in doc :
        TDM[word2id[token], i] += 1    # 해당 토큰의 위치(column)
# 행렬로 표기 ( BOW와 차이점 : BOW는 1차원 배열 )
TDM


# In[60]:


import pandas as pd
doc_names = ['문서' + str(i) for i in range(len(doc_ls))]
sorted_vocab = sorted((value,key) for key, value in word2id.items())

vocab = [v[1] for v in sorted_vocab]
df_TDM = pd.DataFrame(TDM, columns = doc_names)
df_TDM['단어'] = vocab
df_TDM.set_index('단어')


# In[61]:


vocab = []
for v in sorted_vocab :
    #print(v,'**')
    vocab.append(v[1])
print('vocab', vocab)


# In[62]:


vocab = []
for v in sorted_vocab :
    #print(v,'**')
    vocab.append(v[1])
# print('vocab', vocab)

# for i in range(len(docs)) :
#     print("문서{} : {}".format(i, docs[i]))
#     ICD.display(pd.DataFrame([BoW_ls[i]],columns =vocab))
#     print("\n\n")
    

TDM = pd.DataFrame(BoW_ls,columns=vocab)
TDM


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




