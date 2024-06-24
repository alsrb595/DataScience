import pandas as pd
import numpy as np
import sys
import math

train_set = sys.argv[1] # 이렇게 sys.argv를 사용하면 터미널에서 입력한 argument들이 여기에 할당이 된다. 
test_set = sys.argv[2]
classify_result = sys.argv[3]

data = pd.read_csv(train_set, sep = '\t', engine = 'python', header = 0)


feature_list = list(data.columns) #header가 data의 column이 된다. 즉, 각 열의 이름을 리스트로 만든 것
c_label = feature_list[-1] # 맨 뒤의 시리즈에 접근, 즉 class label 
feature_list.pop() # 주어진 data에서 맨 마지막은 class 이름이지 feature가 아니기 때문에 제거
length = len(data)

c_kind = data[c_label].unique().tolist() # class label의 시리즈에서 class 이름들을 리스트로 저장

def Entropy(data_set):
    c, cnt = np.unique(data_set[c_label], return_counts = True)
    entropy = 0

    entropy = -np.sum((cnt / np.sum(cnt)) * np.log2(cnt / np.sum(cnt)))#백터화 연산의 일종임 배열이나 데이터 세트에 대한 연산을 하나의 명령어로 처리하는 기법, 일반적으로 루프, 조건문 등을 사용하지 않고 전체 데이터에 대해 연산을 수행할 수 있게 해줍니다.

    return entropy


def gain(data, feature): #feature를 인자로 받아 데이터를 split하고 entropy 호출을 통해 feature로 split했을 때의 entoropy를 구해 gain을 계산한다. 
    total_entropy = Entropy(data)
    f_entropy = 0
    feature_vlaue, cnt= np.unique(data[feature], return_counts = True) # featur_value 리스트에는 feature가 가지는 고유한 값들이 리스트로 저장되어 있고 cnt 리스트에는 각 고유값들의 등장횟수가 저장되어 있다. 
    for i in range(len(feature_vlaue)):
        f_entropy += (cnt[i] / np.sum(cnt)) * Entropy(data.where(data[feature] == feature_vlaue[i]).dropna()) #노션에 자세히 적어놓음
    gain = total_entropy - f_entropy

    return gain  

def Split_info(data, feature):
    feature_vlaue, cnt= np.unique(data[feature], return_counts = True)
    split_info = 0
    for i in range(len(feature_vlaue)):
        split_info -= cnt[i] / np.sum(cnt) * np.log2(cnt[i] / np.sum(cnt))

    return split_info

def select_best_feature(data, features):
    best_gainratio = 0
    best_feature = None

    for feature in features:
        info_gain = gain(data, feature)
        if info_gain == 0:  # 정보 이득이 0이면 분할 정보 값을 계산하지 않고 넘어감
            continue
        split_info = Split_info(data, feature)
        if split_info == 0:  # 분할 정보가 0이면 이 특성으로는 분할할 수 없음을 의미
            continue
        gain_ratio = info_gain / split_info
        if(gain_ratio > best_gainratio):
            best_gainratio = gain_ratio
            best_feature = feature
    

    return best_feature

def build_decision_tree(data, features):
    # 기저 조건: 모든 데이터가 같은 클래스에 속하거나 특성이 더 이상 없으면 멈춤
    if len(data[c_label].unique()) == 1: #.unique() 사용시 중복값들에 대한 고유값들로 이루어진 넘파이 배열로 반환해줌
        return {'Class_label': data[c_label].iloc[0]}  # 리프 노드의 레이블 반환, 한 노드 안에 모두 같은 class가 들어 있는 것이므로 data[c_label].iloc[0]이 class label 값이 된다. 
    if len(features) == 0:
        # 가장 많이 나타나는 클래스 반환
        return {'Class_label': data[c_label].mode()[0]} # 주어진 data[c_label]에서 .mode() 함수를 이용해 최빈값을 반환한다. 
    if len(data[c_label]) < 5: # 하나의 노드에 5개 이하가 있다면 더 쪼개는게 더 안좋을 수도 있어서 제한을 둠 
        return {'Class_label': data[c_label].mode()[0]}
    
    best_feature = select_best_feature(data, features)

    if best_feature is None:
        return {'Class_label': data[c_label].mode()[0]} 
    
    tree = {best_feature: {}}
    unique_values = data[best_feature].unique()
    new_features = features.copy()
    new_features.remove(best_feature)

    for value in unique_values: #best_feature의 모든 고유값들에 대해 반복
        subset = data[data[best_feature] == value]
        # 해당 분할에 대해 재귀적으로 트리 구축
        tree[best_feature][value] = build_decision_tree(subset, new_features) # [best_feature][value] 딕너리의 키 값으로 사용 된다. [best_feature][value]에서 best_feature는 바깥쪽 딕셔너리의 키 값을 나타내며, value는 해당 best_feature에 대한 내부 딕셔너리의 키 값을 나타냅니다. 
        # best_feature 중 특정 value를 가지는 subset(split_data)을 생성 그리고 그 subset을 다시 build_decision_tree() 함수의 인자로 전달해 트리를 만드는 과정을 재귀적으로 반복 {'age': {'<=30': {'student': {'no': {'label': 'no'}, 'yes': {'label': 'yes'}}} 이런식으로 나옴
    return tree

def predict(tree, data_point):
    if 'Class_label' in tree:  # 리프 노드에 도달한 경우
        return tree['Class_label'] # Class_label을 키 값으로 가지는 value를 반환함
    for feature, subtree in tree.items(): #.item()은 딕셔너리 자료형을 순회할 떄 사용 key, value 값을 반환한다. 여기서는 feature가 key, subtree가 value를 나타낸다. 
        feature_value = data_point[feature] # 특정 행의 feature에 해당하는 feature 값은 무엇인지 확인
        if feature_value in subtree:  # 해당 feature의 값에 해당하는 서브트리가 존재하는 경우
            return predict(subtree[feature_value], data_point)  # 재귀적으로 서브트리 탐색, feature_value를 root로 하는 subtree를 다시 predict의 인자로 넘겨주어 재귀 탐색
    return None  # 트리에 맞는 경로가 없는 경우(None 반환)


# 최초의 트리 구축을 시작합니다.
features = feature_list.copy()
decision_tree = build_decision_tree(data, features)

# 예측 결과를 저장할 리스트 생성
predictions = []

test_data = pd.read_csv(test_set, sep = '\t', engine = 'python', header = 0)

# 테스트 데이터셋의 각 인스턴스에 대해 예측 수행
for _, row in test_data.iterrows():
    prediction = predict(decision_tree, row)
    predictions.append(prediction)

# 예측 결과를 classify_result 파일에 저장
with open(classify_result, 'w') as file:
    for i in feature_list:
        file.write(f"{i}\t")
    file.write(f"{c_label}\n")
    for index, row in test_data.iterrows():
        for j in feature_list:
            file.write(f"{row[j]}\t")
        file.write(f"{predictions[index]}\n")





    
