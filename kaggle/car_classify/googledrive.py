#!/usr/bin/env python
# coding: utf-8

# In[22]:


from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools


# In[23]:


scopes = 'https://www.googleapis.com/auth/drive.file'


# In[24]:


store = file.Storage('storage.json')
creds = store.get()

try :
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None


# In[25]:


if not creds or creds.invalid:
    print('make new cred')
    flow = client.flow_from_clientsecrets('client_drive.json', scopes)
    creds = tools.run_flow(flow, store, flags) if flags else tools.run_flow(flow, store)


# In[ ]:




