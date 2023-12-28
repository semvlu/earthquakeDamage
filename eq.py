import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier     # randomforest = bagging + decision tree
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# read_csv
df = pd.read_csv("train_values.csv")
 
'''
  print(df.isnull().sum()) # check if null value exists
print(df[''].describe())    # Feature info (data distrib)
for i in df:
    df[i].plot.box()
    plt.show()              # check outliers
'''

lbl = pd.read_csv("train_labels.csv")
print(lbl.isnull().sum())
'''
lbl.damage_grade.value_counts().plot.bar(title='Number of building with each damage grade')
plt.show()
selected_features = ['age', 'area_percentage', 'height_percentage', 'count_families']
train_values_subset = df[selected_features]
sns.pairplot(train_values_subset.join(lbl), 
             hue='damage_grade')
plt.show()
'''

test_value = pd.read_csv("test_values.csv")

'print(test_value.isnull().sum())    # check if null value exists'


# change categorical variables into dummy/indicator variables
df = pd.get_dummies(df, columns=['land_surface_condition', 'foundation_type',
'roof_type', 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration',
'legal_ownership_status'])
test_value = pd.get_dummies(test_value, columns=['land_surface_condition', 'foundation_type',
'roof_type', 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration',
'legal_ownership_status'])

# combine 'has_secondary_use' & 'has_secondary_use_APPLICATION' as 1 column
# if 'has_secondary_use_APPLICATION' is 1, change 'has_secondary_use' into 2 to 11, depends on its application type
# so only 0, 2~11 will exist, 1 will NOT exist
# cnt=0 # test for warning ignorance
def change_has_sec_use(data):
    cpy = data['has_secondary_use'].copy() # ignore warning
    for i in range(data.shape[0]):
        if data['has_secondary_use'][i] == 1:
            # cnt+=1
            if data['has_secondary_use_agriculture'][i] == 1:
                cpy[i] = 2
            elif data['has_secondary_use_hotel'][i] == 1:
                cpy[i] = 3
            elif data['has_secondary_use_rental'][i] == 1:
                cpy[i] = 4
            elif data['has_secondary_use_institution'][i] == 1:
                cpy[i] = 5
            elif data['has_secondary_use_school'][i] == 1:
                cpy[i] = 6
            elif data['has_secondary_use_industry'][i] == 1:
                cpy[i] = 7
            elif data['has_secondary_use_health_post'][i] == 1:
                cpy[i] = 8
            elif data['has_secondary_use_gov_office'][i] == 1:
                cpy[i] = 9
            elif data['has_secondary_use_use_police'][i] == 1:
                cpy[i] = 10
            elif data['has_secondary_use_other'][i] == 1:
                cpy[i] = 11
    data['has_secondary_use'] = cpy
    # drop 'has_secondary_use_APPLICATION' columns since their info has been saved into 'has_secondary_use'
    data = data.drop(columns=['has_secondary_use_agriculture','has_secondary_use_hotel',
    'has_secondary_use_rental', 'has_secondary_use_institution','has_secondary_use_school',
    'has_secondary_use_industry', 'has_secondary_use_health_post', 'has_secondary_use_gov_office',
    'has_secondary_use_use_police', 'has_secondary_use_other'])

change_has_sec_use(df)
change_has_sec_use(test_value)

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

'''
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

fig,ax= plt.subplots(4, 3) 

for i in range(4):
    for j in range(3):
        if(3*i+j < 11):
            ax[i,j].plot(info[3*i+j])
plt.show()
'''

# drop building_id
building_id = test_value['building_id']     # for submission.csv
df = df.drop(['building_id'], axis=1)
lbl = lbl.drop(['building_id'], axis=1)
test_value = test_value.drop(['building_id'], axis=1)
lbl = lbl.values.ravel()    # avoid dataConversionWarning
lbl = pd.DataFrame(lbl)

#test
gs = make_pipeline(RobustScaler(), RandomForestClassifier(random_state=42, n_estimators=60, max_depth=15, min_samples_leaf=5, class_weight="balanced"))   
# cascading multi models, standardscaler normalising (Gaussian distrib)

# Assume 'X' is your feature matrix and 'y' is your target variable
# X_train, X_test, y_train, y_test will be the resulting datasets
X_train, X_test, y_train, y_test = train_test_split(df, lbl, test_size=0.2, random_state=42)

gs.fit(df, lbl)
from sklearn.metrics import f1_score

# create submission.csv
predictions = gs.predict(X_test)

f1 = f1_score(y_test,predictions,average=None)
print(f'F1-score: {f1}')
from sklearn.metrics import accuracy_score,recall_score

accuracy = accuracy_score(predictions, y_test)
print(f"Accuracy: {accuracy}")

recall = recall_score(predictions, y_test,average=None)
print(f"Recall: {recall}")


# create submission.csv
predictions = gs.predict(test_value)
submission = pd.DataFrame(data=predictions, columns=['damage_grade'])
submission.insert(0, 'building_id', building_id)
submission.to_csv("submission.csv", index=False)
