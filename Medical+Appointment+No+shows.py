
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd

data = pd.read_csv("/Users/rahmi/Desktop/KDD/KaggleV2-May-2016.csv")
data.head(5)


# In[2]:

data.shape


# In[3]:

data.info()


# In[4]:

data.PatientId.value_counts()


# This shows that many people have scheduled an appointment more than once with the doctor.This might be due to their health issues

# In[5]:

data["Gender"].value_counts()


# In[6]:

import seaborn as sns
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

table_count = data['Gender'].value_counts()
type_index = table_count.index
type_values = table_count.values

fig,ax = plt.subplots(figsize=(6,6))
sns.barplot(x = type_index,y=type_values,ax=ax)
plt.title('Appointments scheduled based on Gender')
plt.xlabel('Gender')
plt.ylabel('Counter')


# This shows that more number of females are scheduling an appointment. This might be because of two reasons
# 1. There are very health conscious and go for regular health checkups
# 2. There immune power is low and are prone to getting diseases.

# In[7]:

#data[(data.PatientId.value_counts() > 1) & (data.Gender == "F")].sum()


# In[8]:

data['year'], data['month'], data['date'] = zip(*data['ScheduledDay'].map(lambda x: x.split('-')))


# In[9]:

data


# In[10]:

data.info()


# In[11]:

table_count = data['year'].value_counts()
type_index = table_count.index
type_values = table_count.values

fig,ax = plt.subplots(figsize=(11,11))
sns.barplot(x = type_index,y=type_values,ax=ax)
plt.title('Appointments scheduled per year')
plt.xlabel('Year')
plt.ylabel('Counter')


# Most of the appointments were made in the year 2016

# In[12]:

table_count = data['month'].value_counts()
type_index = table_count.index
type_values = table_count.values

fig,ax = plt.subplots(figsize=(6,6))
sns.barplot(x = type_index,y=type_values,ax=ax)
plt.title('Appointments scheduled per month')
plt.xlabel('month')
plt.ylabel('Counter')


# In[13]:

data.month.value_counts()


# We can see that in the month of May the appointments are made more. Is it because the spread of diseases more during that season?

# In[14]:

data.date[1]


# In[15]:

data['day'], data['time'] = zip(*data['date'].map(lambda x: x.split('T')))


# In[16]:

data


# In[17]:

table_count = data['day'].value_counts()
type_index = table_count.index
type_values = table_count.values

fig,ax = plt.subplots(figsize=(11,11))
sns.barplot(x = type_index,y=type_values,ax=ax)
plt.title('Appointments scheduled per day')
plt.xlabel('day')
plt.ylabel('Counter')


# In[18]:

data.day.value_counts()


# The plot shows that starting of the month many people are making appointments

# In[19]:

data['hour'], data['minutes'], data['seconds'] = zip(*data['time'].map(lambda x: x.split(':')))


# In[20]:

table_count = data['hour'].value_counts()
type_index = table_count.index
type_values = table_count.values

fig,ax = plt.subplots(figsize=(11,11))
sns.barplot(x = type_index,y=type_values,ax=ax)
plt.title('Appointments scheduled per day')
plt.xlabel('hour of the day')
plt.ylabel('Counter')


# Mostly during early morning people are preferring to consult the doctor

# In[21]:

data.Age.value_counts()


# In[22]:

data[data['Age']<0] 


# This particular record has age less than zero which is not possible

# In[23]:

#Dropping that record
data = data.drop(data[data['Age']<0].index)


# In[24]:

data[data['Age']<0] 


# In[25]:

data['elderly'] = np.where(data['Age']>=50, 'yes', 'no')


# In[26]:

data['Infants'] = np.where(data['Age']<=5, 'yes', 'no')


# In[27]:

table_count = data['elderly'].value_counts()
type_index = table_count.index
type_values = table_count.values

fig,ax = plt.subplots(figsize=(6,6))
sns.barplot(x = type_index,y=type_values,ax=ax)
plt.title('Elderly People')
plt.xlabel('Elderly or not')
plt.ylabel('Counter')


# In[28]:

data.elderly.value_counts()


# In[29]:

data.Infants.value_counts()


# In[30]:

table_count = data['Infants'].value_counts()
type_index = table_count.index
type_values = table_count.values

fig,ax = plt.subplots(figsize=(6,6))
sns.barplot(x = type_index,y=type_values,ax=ax)
plt.title('Infants ')
plt.xlabel('Infants or not')
plt.ylabel('Counter')


# In[31]:

#calculated the middle aged population count by subtracting elderly and Infacts from total population
count_middleaged = 110526 - (11731 + 37036 )
count_middleaged


# It seems like people of age 6 to 49 are planning to visit a doctor more than elderly

# In[32]:

place = data.Neighbourhood.value_counts()
place


# In[33]:

table_count = place.iloc[0:10]
type_index = table_count.index
type_values = table_count.values

fig,ax = plt.subplots(figsize=(15,15))
sns.barplot(x = type_index,y=type_values,ax=ax)
plt.title('Places people are living in')
plt.xlabel('places')
plt.ylabel('Counter')


# In these places people are planning to visit the doctors high 

# In[34]:

data.info()


# In[35]:

table_count = data['Scholarship'].value_counts()
type_index = table_count.index
type_values = table_count.values

fig,ax = plt.subplots(figsize=(5,5))
sns.barplot(x = type_index,y=type_values,ax=ax)
plt.title('Scholarship')
plt.xlabel('Scholarship')
plt.ylabel('Counter')


# To my knowledge on this concept very little people got assistance from govt or NGO for their expenses

# In[36]:

table_count = data['Hipertension'].value_counts()
type_index = table_count.index
type_values = table_count.values

fig,ax = plt.subplots(figsize=(5,5))
sns.barplot(x = type_index,y=type_values,ax=ax)
plt.title('Hipertension')
plt.xlabel('Hipertension')
plt.ylabel('Counter')


# In[37]:

table_count = data['Diabetes'].value_counts()
type_index = table_count.index
type_values = table_count.values

fig,ax = plt.subplots(figsize=(5,5))
sns.barplot(x = type_index,y=type_values,ax=ax)
plt.title('Diabetes')
plt.xlabel('Diabetes')
plt.ylabel('Counter')


# In[38]:

table_count = data['Alcoholism'].value_counts()
type_index = table_count.index
type_values = table_count.values

fig,ax = plt.subplots(figsize=(5,5))
sns.barplot(x = type_index,y=type_values,ax=ax)
plt.title('Alcoholism')
plt.xlabel('Alcoholism')
plt.ylabel('Counter')


# In[39]:



type_index = table_count.index
type_values = table_count.values

fig,ax = plt.subplots(figsize=(5,5))
sns.barplot(x = type_index,y=type_values,ax=ax)
plt.title('Handcap')
plt.xlabel('Handcap')
plt.ylabel('Counter')


# In[40]:

table_count


# In[41]:

table_count = data['SMS_received'].value_counts()
type_index = table_count.index
type_values = table_count.values

fig,ax = plt.subplots(figsize=(5,5))
sns.barplot(x = type_index,y=type_values,ax=ax)
plt.title('SMS_received')
plt.xlabel('SMS_received')
plt.ylabel('counter')


# Number of people who are reminded about their appointment is less. There is a chance of forgetting about their appointment. So this has to be taken care of.

# In[42]:

#calculating the difference between the appointment made and actual appointment date
data['Appointment_year'], data['Appointment_month'], data['Appointment_date'] = zip(*data['AppointmentDay'].map(lambda x: x.split('-')))


# In[43]:

data['Appointment_day'], data['Appointment_time'] = zip(*data['Appointment_date'].map(lambda x: x.split('T')))


# In[44]:

data.info()


# In[45]:

result = data.drop(['Appointment_date','date','Appointment_time'], axis=1)


# In[46]:

result.info()


# In[47]:

result['a_date'] = result[result.columns[23:26]].apply(lambda x: '-'.join(x.dropna().astype(int).astype(str)),axis=1)


# In[48]:

result


# In[49]:

result.info()


# In[50]:

result['s_date'] = result[result.columns[14:17]].apply(lambda x: '-'.join(x.dropna().astype(int).astype(str)),axis=1)


# In[51]:

result


# In[52]:

l_date = pd.to_datetime(result['a_date'])


# In[53]:

f_date = pd.to_datetime(result['s_date'])


# In[54]:

no_days = l_date - f_date


# In[55]:

result['no_days'] = no_days


# In[56]:

result


# In[57]:

var_days = result['no_days'].astype('timedelta64[s]')

plt.hist(var_days)
plt.title("Difference b/w appointment and scheduled date")
plt.xlabel("Value")
plt.ylabel("Frequency")


# In[58]:

#seeing how each variable varies with target variable
result['No-show'].value_counts()


# In[171]:

def probStatus(dataset, group_by):
    df = pd.crosstab(index = dataset[group_by], columns = dataset['No-show']).reset_index()
    df['probShowUp'] = df['Yes'] / (df['No'] + df['Yes'])
    return df[[group_by, 'probShowUp']]


# In[172]:

sns.lmplot(data = probStatus(result, 'Age'), x = 'Age', y = 'probShowUp', fit_reg = True)
sns.plt.xlim(0, 100)
sns.plt.title('Probability of showing up with respect to Age')
sns.plt.show()


# In[59]:

result.info()


# In[60]:

result['Gender'] = np.where(result['Gender'] == 'F', 1 , 0 )


# In[61]:

result


# In[62]:

#normalising age
age_minmax = pd.DataFrame(result['Age'])
age_minmax = age_minmax.apply(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)), axis = 0)
result['age_minmax'] = age_minmax


# In[63]:

s = result['no_days'].astype('timedelta64[s]')

days_minmax = pd.DataFrame(s)
days_minmax = days_minmax.apply(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)), axis = 0)
result['days_minmax'] = days_minmax


# In[64]:

#correlation
corrmat =result.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()


# In[145]:

#Modelling
#'Gender','age_minmax','Scholarship','Hipertension','Diabetes','Alcoholism','Handcap','SMS_received','days_minmax'
attributes = result[['days_minmax','age_minmax' ]]
result['Attack'] = result['No-show'].map({'No': 0, 'Yes': 1})
target = result['Attack']


# In[146]:

from sklearn.cross_validation import train_test_split
x_train, x_val, y_train, y_val = train_test_split(attributes, target,
                                                  test_size = .2,
                                                  random_state=42)


# In[147]:

from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)


# In[148]:

#Modeling
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)
clf_rf.fit(x_train_res, y_train_res)


# In[149]:

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_predicted = clf_rf.predict(x_val)
conf_matrix = confusion_matrix(y_val, y_train_predicted)
conf_matrix


# In[150]:

print('\nTest Results')
print('Accuracy',clf_rf.score(x_val, y_val))


# In[151]:

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_val,y_train_predicted)


# In[152]:

def plot_roc_curve(fpr, tpr, label = None):
    plt.plot(fpr, tpr, linewidth=2, label= label)
    plt.plot([0,1],[0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False positive Rate')
    plt.ylabel('True Postive Rate')
plot_roc_curve(fpr,tpr)
plt.show()


# In[153]:

from sklearn.metrics import roc_auc_score
roc_auc_score(y_val,y_train_predicted)


# In[154]:

print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), clf_rf.feature_importances_), attributes), 
             reverse=True))


# In[ ]:



