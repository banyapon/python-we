import json
import requests
from firebase import firebase
firebase = firebase.FirebaseApplication('https://enet5-7f9f6.firebaseio.com/', None)
result = firebase.get('/person/', '')
for person in result:
    jsn = requests.get('https://enet5-7f9f6.firebaseio.com/person.json')
    data = jsn.json()
    print(data[person]['Name'])
    print(data[person]['Surname'])
    print(data[person]['Image'])