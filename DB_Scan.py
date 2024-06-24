import pandas as pd
import numpy as np
import sys

input_data = sys.argv[1]
n = int(sys.argv[2])
eps = float(sys.argv[3])
minpts = int(sys.argv[4])

data = pd.read_csv(input_data, sep = "\t", engine = "python", encoding = "cp949", header = None, names = ["id", "x", "y", "label"])
data["label"] = np.nan


def RangeQuery(data, dp, eps):
    cur_dp = np.array([data.iloc[dp]["x"], data.iloc[dp]["y"]]) 
    eu_dis = np.sqrt(np.sum((data[["x", "y"]].values - cur_dp) ** 2)) # data[["x", "y"]].values x와 y 컬럽의 데이터를 가진 이차원 배열이 생성된다. 
    #2차원 배열에 대한 np.sum() 이고 axis = 1로 지정했기 때문에 각 행에 대한 계산 결과를 배열로 저장을 한다. 
    #넘파이의 브로드캐스팅 기능 덕분에 이 계산은 data의 모든 점에 대해 수행된다. 
    return np.where(eu_dis <= eps)[0]
 #eu_dis 자체가 유클리디안 거리 데이터를 가지고 있는 넘파이 배열 그 자체임 [0]은 인덱스 번호 데이터를 가진 컬럼을 선택하는 것
    
    
            
c_num = 0

for index, row in data.iterrows():
    if c_num == n:   
        break
    if pd.notna(row["label"]):
        continue 
    
    N = set(RangeQuery(data, index, eps))
    

    if len(N) < minpts:
        data.at[index, "label"] = "Noise"
        continue

    data.at[index, "label"] = c_num

    N.remove(index)

    seed_set = N.copy()

    while seed_set:
        dp = seed_set.pop() #pop() 하면서 pop() 하는 갑을 반환하는 기능도 있음
        dp_label = data.iloc[dp]["label"]
        if dp_label == "Noise":
            data.at[dp, "label"] = c_num
        if pd.notna(dp_label): #Nan 값인지 확인하는 함수 notna()
            continue

        data.at[dp, "label"] = c_num        
        
        sub_N = set(RangeQuery(data, dp, eps))

        if len(sub_N) < minpts:
            continue

        seed_set = seed_set.union(sub_N) # seed set을 계속 추가를 해줘야 됨 추가를 안해주면 clustering 에서 빠지는 데이터가 존재할 수도 있다. 
        sub_N.clear()
    c_num += 1

clusters = data[data['label'] != 'Noise'].groupby('label')
for label, cluster in clusters: # label은 그룹화된 키 값 즉, label열에서 고유한 값들을 의미한다. 
    output_file = input_data.replace(".txt", "") + "_cluster_" + str(int(label)) + ".txt"
    with open(output_file, "w") as file:
        for index in cluster.index:
            file.write(str(index) + "\n")

