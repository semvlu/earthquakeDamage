import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt

df = pd.read_csv("train_values.csv")
# change categorical variables into dummy/indicator variables
df = pd.get_dummies(df, columns=['land_surface_condition', 'foundation_type',
'roof_type', 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration',
'legal_ownership_status'])

# combine 'has_secondary_use' & 'has_secondary_use_APPLICATION' as 1 column
# if 'has_secondary_use_APPLICATION' is 1, change 'has_secondary_use' into 2 to 11, depends on its application type
# so only 0, 2~11 will exist, 1 will NOT exist
cnt=0 # test for warning ignorance
for i in range(df.shape[0]):
    if df['has_secondary_use'][i] == 1:
        cnt+=1
        if df['has_secondary_use_agriculture'][i] == 1:
            df['has_secondary_use'][i] = 2
        elif df['has_secondary_use_hotel'][i] == 1:
            df['has_secondary_use'][i] = 3
        elif df['has_secondary_use_rental'][i] == 1:
            df['has_secondary_use'][i] = 4
        elif df['has_secondary_use_institution'][i] == 1:
            df['has_secondary_use'][i] = 5
        elif df['has_secondary_use_school'][i] == 1:
            df['has_secondary_use'][i] = 6
        elif df['has_secondary_use_industry'][i] == 1:
            df['has_secondary_use'][i] = 7
        elif df['has_secondary_use_health_post'][i] == 1:
            df['has_secondary_use'][i] = 8
        elif df['has_secondary_use_gov_office'][i] == 1:
            df['has_secondary_use'][i] = 9
        elif df['has_secondary_use_use_police'][i] == 1:
            df['has_secondary_use'][i] = 10
        elif df['has_secondary_use_other'][i] == 1:
            df['has_secondary_use'][i] = 11

# drop 'has_secondary_use_APPLICATION' columns since their info has been saved into 'has_secondary_use'
df = df.drop(columns=['has_secondary_use_agriculture','has_secondary_use_hotel',
'has_secondary_use_rental', 'has_secondary_use_institution','has_secondary_use_school',
'has_secondary_use_industry', 'has_secondary_use_health_post', 'has_secondary_use_gov_office',
'has_secondary_use_use_police', 'has_secondary_use_other'])
# drop n/a values
df = df.dropna()
# test if warning can be ignored
"""
count=[]
for i in range(2,12):
    count.append (df['has_secondary_use'].value_counts()[i])
sum=0
for i in range(len(count)):
    sum+=count[i]
print(cnt)
print(sum)
"""
lbl = pd.read_csv("train_labels.csv")
info = []
for i in range(11):
    info.append([0,0,0])

for i in range(df.shape[0]):
    if df['has_secondary_use'][i]==0:
        if lbl['damage_grade'][i] == 1:
            info[0][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[0][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[0][2]+=1
    elif df['has_secondary_use'][i]==2:
        if lbl['damage_grade'][i] == 1:
            info[1][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[1][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[1][2]+=1
    elif df['has_secondary_use'][i]==3:
        if lbl['damage_grade'][i] == 1:
            info[2][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[2][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[2][2]+=1

    elif df['has_secondary_use'][i]==4:
        if lbl['damage_grade'][i] == 1:
            info[3][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[3][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[3][2]+=1

    elif df['has_secondary_use'][i]==5:
        if lbl['damage_grade'][i] == 1:
            info[4][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[4][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[4][2]+=1
    
    elif df['has_secondary_use'][i]==6:
        if lbl['damage_grade'][i] == 1:
            info[5][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[5][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[5][2]+=1

    elif df['has_secondary_use'][i]==7:
        if lbl['damage_grade'][i] == 1:
            info[6][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[6][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[6][2]+=1

    elif df['has_secondary_use'][i]==8:
        if lbl['damage_grade'][i] == 1:
            info[7][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[7][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[7][2]+=1

    elif df['has_secondary_use'][i]==9:
        if lbl['damage_grade'][i] == 1:
            info[8][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[8][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[8][2]+=1

    elif df['has_secondary_use'][i]==10:
        if lbl['damage_grade'][i] == 1:
            info[9][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[9][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[9][2]+=1
    elif df['has_secondary_use'][i]==11:
        if lbl['damage_grade'][i] == 1:
            info[10][0]+=1
        elif lbl['damage_grade'][i] == 2:
            info[10][1]+=1
        elif lbl['damage_grade'][i]==3:
            info[10][2]+=1

for i in info:
    plt.plot(i)
plt.show()
