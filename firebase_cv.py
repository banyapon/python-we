from firebase import firebase

firebase = firebase.FirebaseApplication('https://<<ID>>.firebaseio.com/', None)
data =  { 'Name': 'BANYAPON',
          'Surname': 'POOLSAWAS',
          'Gender': 'M'
          }
result = firebase.post('/person',data)
print(result)