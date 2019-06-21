
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_100e2d4c6bf04f36aff48a66692f404d = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='oOSIoNz9t8qFUsFLrnpNO4pZxxeYhupte62tcrARZorf',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_100e2d4c6bf04f36aff48a66692f404d.get_object(Bucket='zomatocustomerreviewprediction-donotdelete-pr-2wbm5djncdmz9d',Key='zomato.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset = pd.read_csv(body)
dataset.head()



# In[3]:


dataset.drop(['Rating_color','Votes','Aggregate_rating'],axis=1,inplace=True)
dataset


# In[4]:


dataset.isnull().any()


# In[5]:


x=dataset.iloc[:,0:5].values


# In[6]:


x


# In[7]:


y=dataset.iloc[:,5:].values


# In[8]:


y


# In[9]:


x.shape


# In[10]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
x[:,1]=lb.fit_transform(x[:,1])
x


# In[11]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
x[:,2]=lb.fit_transform(x[:,2])
x


# In[12]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
x[:,3]=lb.fit_transform(x[:,3])
x


# In[13]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
y[:,0]=lb.fit_transform(y[:,0])
y


# In[14]:


from sklearn.preprocessing import OneHotEncoder
oh=OneHotEncoder(categorical_features=[0])
y=oh.fit_transform(y).toarray()
y


# In[15]:


y.shape


# In[16]:


y=y[:,1:]
y


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_test


# In[18]:


x_train.shape


# In[19]:


y_train.shape


# In[20]:


y_train


# In[21]:


y_test.shape


# In[22]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=30,criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)


# In[23]:


y_predict=classifier.predict(x_test)
y_predict


# In[24]:


y_predict=classifier.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# In[25]:


get_ipython().system(u'pip install watson-machine-learning-client --upgrade')


# In[26]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient 


# In[27]:


wml_credentials={
  "instance_id": "76cf34f1-a6bf-496d-a074-e946190910b6",
  "password": "d93398f4-531e-4f3d-8573-d472d1946528",
  "url": "https://eu-gb.ml.cloud.ibm.com",
  "username": "ca6dbb42-0c67-4243-b65e-96809253273a",
  "acces_key": "KEBBC0veMU18fy47kUbBkMMB9NwUfH4mXBonz3spJirf"}


# In[28]:


client=WatsonMachineLearningAPIClient(wml_credentials)


# In[29]:


import json
instance_details=client.service_instance.get_details()
print(json.dumps(instance_details,indent=2))


# In[30]:


model_props={client.repository.ModelMetaNames.AUTHOR_NAME:"Srilaxmi",
            client.repository.ModelMetaNames.AUTHOR_EMAIL:"srilaxmigopala971@gmail.com",
            client.repository.ModelMetaNames.NAME:"Randomforest"}


# In[32]:


model_artifact=client.repository.store_model(classifier,meta_props=model_props)


# In[33]:


published_model_uid=client.repository.get_model_uid(model_artifact)


# In[34]:


published_model_uid


# In[35]:


created_deployment=client.deployments.create(published_model_uid,name="Randomforest")


# In[36]:


scoring_endpoint=client.deployments.get_scoring_url(created_deployment)
scoring_endpoint


# In[37]:


client.deployments.list()

