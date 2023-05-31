#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')


# In[3]:


match.head()


# In[4]:


match.shape


# In[5]:


delivery.head()


# This is basically a classifically problem but the outcome that we have to show should be in percentages or in probabilities so we are selecting those classification problems that show the probability like Logistic regression,support vector machine

# In[6]:


total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()


# In[7]:


total_score_df


# In[8]:


total_score_df = total_score_df[total_score_df['inning']==1]


# filtering the data
# here we are only taking into consideration the scores of the team batting first as we will be predicting the win or loss probability while the second team is batting

# In[9]:


total_score_df


# In[10]:


match_df = match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')


# # Data Cleaning and Wrangling

# In[11]:


match_df['team1'].unique()


# In[12]:


teams=['Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
       'Kolkata Knight Riders','Delhi Capitals','Kings XI Punjab','Rajasthan Royals','Chennai Super Kings']


# In[13]:


match_df['team1']=match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2']=match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1']=match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2']=match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')


# In[14]:


match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]


# In[15]:


match_df.shape


# In[16]:


match_df.head()


# In[17]:


match_df['dl_applied'].value_counts()


# In[18]:


match_df = match_df[match_df['dl_applied']==0]


# In[19]:


match_df.columns


# In[20]:


match_df = match_df[['match_id','city','winner','total_runs']]


# In[21]:


delivery_df = match_df.merge(delivery,on='match_id')


# In[22]:


delivery_df = delivery_df[delivery_df['inning']==2]


# In[23]:


delivery_df.shape


# In[24]:


delivery_df.head()


# # Feature Engineering

# In[25]:


delivery_df['current_score'] = delivery_df.groupby('match_id').cumsum()['total_runs_y']


# In[26]:


delivery_df['runs_left'] = delivery_df['total_runs_x']-delivery_df['current_score']+1


# In[27]:


delivery_df.head()


# # Balls Left

# In[28]:


delivery_df['balls_left'] = 126 - (delivery_df['over']*6+delivery_df['ball'])


# In[29]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x if x == "0" else "1")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
wickets = delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
delivery_df['wickets'] = 10 - wickets
delivery_df.head()


# # Current run rate

# In[30]:


delivery_df['crr'] = (delivery_df['current_score']*6)/(120-delivery_df['balls_left'] )


# # Required run rate

# In[31]:


delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']


# In[32]:


def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0


# In[33]:


delivery_df['result'] = delivery_df.apply(result,axis=1)


# In[34]:


delivery_df.head()


# In[35]:


final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]


# In[36]:


final_df.head()


# We are shuffling because all the balls of the match are continuous so bias may develop 

# In[37]:


final_df = final_df.sample(final_df.shape[0])


# In[38]:


final_df.sample()


# In[39]:


final_df.dropna(inplace=True)


# In[40]:


final_df = final_df[final_df['balls_left'] != 0]


# # Model Training

# In[41]:


X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# We are applying OneHotEncoding on batting_team, bowling_team and vejnue as they are in string format

# In[42]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')


# sparese=False as we dont need in array format

# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[44]:


pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])


# In[45]:


pipe.fit(X_train,y_train)


# In[46]:


y_pred = pipe.predict(X_test)


# In[47]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100


# # Testing and Prediction

# In[48]:


final_df.iloc[[74]]


# In[49]:


pipe.predict_proba(X_test)[108]*100


# In[50]:


def match_summary(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total_runs_x']))


# In[51]:


def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target


# In[52]:


temp_df,target = match_progression(delivery_df,74,pipe)
temp_df


# In[53]:


import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))

