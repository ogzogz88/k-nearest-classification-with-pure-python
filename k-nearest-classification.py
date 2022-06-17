#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import related libraries
# --------------------------------------------------------------------------------------------------
# ilgili kütüphaneleri import et

import pandas as pd
import numpy as np
import random as rnd
import statistics
import math
from collections import Counter

# show all rows
# --------------------------------------------------------------------------------------------------
# veri setinin tüm satırlarını görüntüle
pd.set_option("max_rows", None)

# load the dataset
# --------------------------------------------------------------------------------------------------
# veri setini yükle
avila_all = pd.read_csv("avila-all.csv")


# In[2]:


# read the data
# --------------------------------------------------------------------------------------------------
# veri setini oku
avila_all.head()


# In[3]:


# see number of rows of the data
# --------------------------------------------------------------------------------------------------
# verinin satır sayısına bak
number_of_rows = len(avila_all)
print("***********")
print("number_of_rows")
print(number_of_rows)
print("***********")


# In[60]:


# https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
# https://numpy.org/doc/stable/reference/generated/numpy.split.html
# numpy.array_split(ary, indices_or_sections, axis=0)
# Split array into multiple sub-arrays of equal size.
# returns a list of ndarrays 

# Split the dataset into n sub-datasets and get the first one.
# --------------------------------------------------------------------------------------------------
# Veri setini eşit sayıda veri içeren n alt veri setine böl, birinciyi al.

# my_df = np.array_split(avila_all, 30)[0]


# In[4]:


# View the sub-dataset.
# --------------------------------------------------------------------------------------------------
# alt veri setini görüntüle.

# my_df


# In[5]:


# k-Nearest Neighbors in 3 steps
# Step 1: Calculate the Distance.
# Step 2: Get k number of nearest neighbors.
# Step 3: Make predictions.
# --------------------------------------------------------------------------------------------------
# 3 adımda k-En Yakın Komşu 
# Adım 1: Uzaklıkları hesapla.
# Adım 2: k adet en yakın komşuyu bul.
# Adım 3: Tahminleme yap.


# In[10]:


# get_canberra_distance function
# returns a list including the k number of nearest classes
# --------------------------------------------------------------------------------------------------
# get_canberra_distance fonksiyonu
# k adet en yakın sınıfı içeren bir liste döndürür

"""
parameters
-----------------
test_df          : test dataframe to use the classes of (pandas DataFrame)
k                : number of nearest neighbors
train_row        : data point to calculate the distances from
d_row_length     : dataframe row length
d_column_length  : dataframe column length
--------------------------------------------------------------------------------------------------
parametreler
-----------------
test_df          : sınıflarını kullanacağımız test veriseti (pandas DataFrame)
k                : en yakın komşu sayısı
train_row        : uzaklık hesaplamasının yapılacağı veri noktası(satır)
d_row_length     : dataframe satır sayısı
d_column_length  : dataframe sütun sayısı
"""

def get_canberra_distance(test_df, train_row, k, d_row_length, d_column_length):
    
    # DataFrame.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
    # Drop specified labels from rows or columns.
    # axis{0 or ‘index’, 1 or ‘columns’}, default 
    # Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
    # --------------------------------------------------------------------------------------------------
    # Sütundaki bir etiketi silmek için axis = 1 
    df_without_class = test_df.drop(['class name'], axis=1)

    distances = []
    
    # pandas.DataFrame.iloc
    # https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html#pandas-dataframe-iloc
    # Purely integer-location based indexing for selection by position.
    # --------------------------------------------------------------------------------------------------
    # Pandas dataframe'den index bazlı veri seçer.   
    for i in range(d_row_length):
        
        # list keeping the distance between two different data points/rows.
        # --------------------------------------------------------------------------------------------------
        # iki farklı nokta arasındaki uzaklığı tutan liste.
        distance_by_row = []
        
        for j in range(d_column_length-1):
            
                
            # Canberra distance formula
            # D=∑ni=1 (|Xi−Yi| / |Xi|+|Yi|)
            # --------------------------------------------------------------------------------------------------
            # Canberra uzaklığı formülü
            # D=∑ni=1 (|Xi−Yi| / |Xi|+|Yi|)
      
            distance = abs(train_row.iloc[0,j] - df_without_class.iloc[i,j])/(abs(train_row.iloc[0,j]) + abs(df_without_class.iloc[i,j]))
            distance_by_row.append(distance)

        distances.append(sum(distance_by_row))
            
    # sort distances with ascending order
    # --------------------------------------------------------------------------------------------------
    # uzaklıkları artan sırada sırala
    sorted_distances = sorted(distances)

    # list including classes of the first k-nearest number of distances
    # --------------------------------------------------------------------------------------------------
    # en yakın k adet komşunun sınıfını içeren list
    first_k_neighbor_classes = []
    
    for index in range(len(sorted_distances)):
        for neighbor_index in range(k):
            
            # get the classnames of first k neighbors and add to the list
            # --------------------------------------------------------------------------------------------------
            # k adet en yakın komşunun sınıf ismini listeye ekle
            if(sorted_distances[neighbor_index] == distances[index]):
                first_k_neighbor_classes.append(test_df.iloc[index, d_column_length-1])
                

    
    return first_k_neighbor_classes
            
        
    
    
        

    
    


# In[11]:


# calculate_confusion_matrix function
# returns and logs a list of objects including confusion matrix info (TP, FP, TN, TP).
# --------------------------------------------------------------------------------------------------
# calculate_confusion_matrix fonksiyonu
# confusion matriks bilgisini (TP, FP, TN, TP) içeren nesnelerden oluşan bir list loglar ve döndürür.

"""
parameters
-----------------
result           : a DataFrame with 2 columns, first column predicted-class and second row real class
--------------------------------------------------------------------------------------------------
parametreler
-----------------
result           : 2 sütunlu bir DataFrame, ilk sütun tahmini sınıflar ve ikinci sütun gerçek sınıflar
"""

def calculate_confusion_matrix(result):
    
    # TP: True Positive, tahmin değeri DOĞRU gerçek değer DOĞRU
    # FP: False Positive, tahmin değeri DOĞRU gerçek değer YANLIŞ
    # TN: True Negative, tahmin değeri YANLIŞ gerçek değer YANLIŞ
    # FN: False Negative, tahmin değeri YANLIŞ gerçek değer DOĞRU
    
    
    # class names in the dataset
    # --------------------------------------------------------------------------------------------------
    # veri setindeki sınıf isimleri
    classes = ["A","B","C","D","E","F","G","H","I","W","X","Y"]
    
    c_len = len(classes)
    r_len = len(result)
    
    confusion_matrix = []
    
    
    for i in range(r_len):        
        for c in range(c_len):
            confusion_matrix_values = {"tp":0, "fp":0, "tn":0, "fn":0} 

            # add confusion matrix values per classname.
            # --------------------------------------------------------------------------------------------------
            # her bir sınıf ismi için confusion matriks verisi ekle.
            if(len(confusion_matrix) == c):
                confusion_matrix.append({classes[c]:confusion_matrix_values})
                
            if(result.iloc[i,0] == classes[c] and result.iloc[i,0] == result.iloc[i,1]):
                confusion_matrix[c][classes[c]]["tp"] += 1
                                    
            elif(result.iloc[i,0] == classes[c] and result.iloc[i,1] != classes[c]):
                confusion_matrix[c][classes[c]]["fp"] += 1 
            
            elif(result.iloc[i,0] != classes[c] and result.iloc[i,1] == classes[c]):
                confusion_matrix[c][classes[c]]["fn"] += 1
                                    
            elif(result.iloc[i,0] != classes[c] and result.iloc[i,1] != classes[c]):
                confusion_matrix[c][classes[c]]["tn"] += 1
    
    print("confusion matrix")
    print(confusion_matrix)
    
    return confusion_matrix          
            
               
    
    
    


# In[12]:


# find_mode function
# takes 1 argument - a list and returns the most occuring item or most occuring items list if there is more than 1 item.
# --------------------------------------------------------------------------------------------------
# find_mode fonksiyonu
# parametre olarak 1 tane list alır, en çok tekrar eden elemanı ya da 1'den fazla ise eleman dizisini döndürür.
def find_mode(mode_list):
    
    c = Counter(mode_list)
    return [k for k, v in c.items() if v == c.most_common(1)[0][1]]


# In[13]:


# k_nearest function
# returns a DataFrame with 2 columns, first column predicted-class and second row real class.
# --------------------------------------------------------------------------------------------------
# k_nearest fonksiyonu
# 2 sütunlu bir DataFrame döndürür, ilk sütun tahmini sınıflar ve ikinci sütun gerçek sınıflar

"""
parameters
-----------------
df   : dataframe (pandas DataFrame).
k    : number of nearest neighbors.

--------------------------------------------------------------------------------------------------
parametreler
-----------------
df   : 2 sütunlu bir DataFrame, ilk sütun tahmini sınıflar ve ikinci sütun gerçek sınıflar.
k    : veriseti (pandas DataFrame).
"""

def k_nearest(df, k):
    
    result = pd.DataFrame(columns=['Predicted-C', 'Real-C'])
    
    d_row_length = len(df)
    d_row_length_1_3 = math.ceil(d_row_length/3)
    d_row_length_2_3 = d_row_length - d_row_length_1_3
    d_column_length = len(df.columns)
    
    print("rows and column lenn")
    print(f"1-3: {d_row_length_1_3}, 2-3:{d_row_length_2_3}, all:{d_row_length}")
    
    
    partition_flag = 0
    while(partition_flag <= 2):
        
        
        train_df = df.iloc[:d_row_length_1_3]
        test_df = df.iloc[d_row_length_1_3:d_row_length]
        
        number_of_true_prediction = 0
        
        # print("train df")
        # print(train_df)
        # print("test df")
        # print(test_df)
        
        for start_index in range(d_row_length_1_3):
            
            train_row = train_df.iloc[[start_index]]
            
            # first k number of neighbor classes.
            # --------------------------------------------------------------------------------------------------
            # ilk k adet komşu sınıf.
            neighbor_classes = get_canberra_distance(test_df, train_row, k, d_row_length_2_3, d_column_length)
        
            # find predicted-class based on mode value of neighbor classes.
            # --------------------------------------------------------------------------------------------------
            # en çok tekrar eden komşu sınıf sayısına göre tahmini sınıfı bul.
            print(f"start-index: {start_index}")
            print(neighbor_classes)
        
            # https://docs.python.org/3/library/statistics.html#statistics.mode
            # https://www.tutorialsteacher.com/python/statistics-module
            # The statistics.mode() method returns the most common data point in the list.
            # if same number of modes, returns the first encountered one in the list.
            # --------------------------------------------------------------------------------------------------
            # statistics.mode() metodu bir list içindeki en çok tekrar eden noktayı döndürür.
            # en çok tekrar eden birden çok eleman var ise, list içindeki ilk karşılaşılan elemanı döndürür.
            # predicted_class = statistics.mode(neighbor_classes)
            
            
            real_class = train_df.iloc[start_index, d_column_length-1]
            
            predicted_class = find_mode(neighbor_classes)[0]
            predicted_len = len(predicted_class)
            
            # if there is more than 1 mode AND if one of them equals real_class, choose real_class value,
            # else choose the first one(we already could not predict correctly, so it does not matter which value to assign).
            # if there is only 1 mode value, we again take predicted_class[0], because find_mode function returns a list.
            # --------------------------------------------------------------------------------------------------
            # eğer 1'den fazla mode değeri varsa VE eğer bunlardan biri gerçek sınıf değerine eşitse, gerçek değeri seç,
            # gerçek sınıf değeri burada yoksa ilk sınıfı ata(zaten doğru tahmin edemedik, ne atadığımız önemli değil).
            # eğer sadece 1 tane mode değeri var ise gene predicted_class[0] değerini alıyoruz, çünkü find_mode fonksiyonu
            # list döndürüyor.
            if(predicted_len > 1):
                for i in range(predicted_len):
                    if(predicted_class[i] == real_class):
                        predicted_class = predicted_class[i]
                        break
                predicted_class = predicted_class[0]
            else:
                predicted_class = predicted_class[0]
                    
        
            # pandas.DataFrame.loc
            # https://pandas.pydata.org/docs/reference/frame.html
            # Access a group of rows and columns by label(s) or a boolean array.
            # --------------------------------------------------------------------------------------------------
            # Satır ve sütunlara etiket adına ya da boolean dizisine göre eriş.
            result.loc[start_index] = [predicted_class, real_class]
            
            # calculate true prediction percentage.
            # --------------------------------------------------------------------------------------------------
            # doğru tahmin oranını hesapla.
            if(result.iloc[start_index,0] == result.iloc[start_index,1]):
                number_of_true_prediction += 1

        calculate_confusion_matrix(result)
    
        # calculate success rate.
        # --------------------------------------------------------------------------------------------------
        # başarı oranını hesapla.
        success_rate = float("{:.2f}".format(number_of_true_prediction / len(result) * 100))
    
        print("**********")
        print(f"success_rate: %{success_rate}")
        print("**********")
        print(result)
        
        # add train_df after test_df and concatenate them so that they change after each iteration.
        # In this way, we will have performed the 3-part cross-validation.
        # --------------------------------------------------------------------------------------------------
        # eğitim verisini test verisinin sonuna ekle ve birleştir. Böylece her iterasyonda bu veriler değişecek.
        # Bu şekilde 3 parçalı çapraz doğrulamayı gerçekleştirmiş olacağız.
        df = pd.concat([test_df, train_df], axis=0)
        
        # reset true prediction number.
        # --------------------------------------------------------------------------------------------------
        # doğru tahmin sayısını sıfırla.
        number_of_true_prediction = 0
        
        partition_flag += 1
    
    print("--------------------------------------------------------------------------------------------------")
    print("FINISHED")
    print("--------------------------------------------------------------------------------------------------")


# In[14]:


# The recommended value for the "k" parameter is the "square root of the number of data" to start with.
# After this initial value, different values should be tried according to the type of data and the area to be studied.
# the optimal "k" value must be found. The most successful results in the study with a small part of the existing data set
# Obtained by taking "half the square root of the data number" as the "k" value.
# The total number of rows of our data is 20867. Since we will do 3-part cross-validation, the data used in the training
# the number of lines (train_df) is 6956. When we take half the square root of this number and round it up, we get k = 43.
# It is recommended that the "k" parameter be an odd number.
# --------------------------------------------------------------------------------------------------
# "k" parametresi için önerilen, başlangıç için "veri sayısının karekökü" değeridir.
# Bu başlangıç değerinden sonra verinin tipi ve çalışma yapılan alana göre farklı değerler denenmeli ve 
# optimal "k" değeri bulunmalıdır. Mevcut veri setinin küçük bir kısmıyla yapılan çalışmada en başarılı sonuçlar
# "k" değeri olarak "veri sayısının karekökünün yarısı" alınarak elde edilmiştir.
# Verimizin toplan satır sayısı 20867'dır. 3 parçalı çapraz doğrulama yapacağımız için eğitimde kullanılan verinin
# satır sayısı (train_df) 6956'dır. Bu sayısının karekökünün yarısını alıp yukarı yuvarladığımızda k = 43 elde ediyoruz.
# Tavsiye edilen "k" parametresinin tek sayı tercih edilmesidir.

# daha az sayıda veriyle deneme yapmak için, veri setini n sayısına (burada 30) bölerek algoritmayı çalıştırabiliriz.
# --------------------------------------------------------------------------------------------------
# Split the dataset into n sub-datasets and get the first one.
# --------------------------------------------------------------------------------------------------
# Veri setini eşit sayıda veri içeren n alt veri setine böl, birinciyi al.
# my_df = np.array_split(avila_all, 30)[0]


k_nearest(avila_all,43)


# In[69]:


k_nearest(my_df,7)


# In[ ]:




