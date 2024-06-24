import pandas as pd
from itertools import combinations
min_sup = 0.1

f = open('./input.txt', 'r')
lines = f.readlines()

max_length = max(len(line.split()) for line in lines)

# 모든 행의 길이를 맞추고, 공백으로 채움
for i, line in enumerate(lines):
    lines[i] = line.rstrip('\n').split('\t') + [''] * (max_length - len(line.split()))

# DataFrame으로 변환
df = pd.DataFrame(lines)

def apriori(df, min_sup):
    C1 = []
    for row in range(len(df)):
        for item in df.iloc[row]:
            if item not in C1 and item != '':
                C1.append(item)

    F = []
    L = []
    support_cnt = [0] * len(C1)

    for i, item in enumerate(C1):
        for index, row in df.iterrows():
            if item in row.values:
                support_cnt[i] += 1
        if support_cnt[i] / len(df) > min_sup:
            F.append({item})
            L.append({item})
    k = 2
    while(L):
        Ck = gen_Ck(L, k)
        L = scan_db(F, Ck, df, min_sup)
        k += 1
    return F

def gen_Ck(L, k):
    Ck = []
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            if i != j and len(L[i] | L[j]) == k:
                Ck.append(L[i] | L[j])
    return Ck

def scan_db(F, Ck, df, min_sup):
    #scan_db
    L = []
    support_cnt = [0] * len(Ck)
    for i, item in enumerate(Ck):
        for index, row in df.iterrows():
            if item.issubset(set(row)):
                support_cnt[i] += 1
                
        if support_cnt[i] / len(df) > min_sup and item not in F:
            F.append(item)
            L.append(item)
    return L

def assosiation(freq_pt, df, min_sup):
    support_cnt = [0] * len(freq_pt)
    for i, item in enumerate(freq_pt):
        for index, row in df.iterrows():
            if item.issubset(set(row)) and len(item) > 1:
                support_cnt[i] += 1

        support = (support_cnt[i] / len(df)) * 100
        all_subset = []
        if support > min_sup * 100:
            for k in range(1, len(item)):
                subsets = combinations(item, k)
                all_subset.extend(subsets)
                for subset in all_subset:
                    subset = set(subset)
                    item1 = item.difference(subset)
                    confi = cal_confi(item1, subset) * 100
                    print(f"{item1}\t{subset}\t{support : .2f}\t{confi : .2f}")
                    print()

def cal_confi(item1, subset):
    cnt1 = 0
    cnt2 = 0
    for index, row in df.iterrows():
        if item1.issubset(set(row)):
            cnt1 += 1
            if item1.issubset(set(row)) and subset.issubset(set(row)):
                cnt2 += 1
    return cnt2 / cnt1


freq_pt = apriori(df, min_sup)

assosiation(freq_pt, df, min_sup)

