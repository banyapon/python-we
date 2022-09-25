import json
import requests
from firebase import firebase
firebase = firebase.FirebaseApplication('https://<<ID>>.firebaseio.com/', None)
result = firebase.get('/person/', '')
for person in result:
    jsn = requests.get('https://<<ID>>.firebaseio.com/person.json')
    data = jsn.json()
    print(data[person]['Name'])
    print(data[person]['Surname'])
    print(data[person]['Image'])