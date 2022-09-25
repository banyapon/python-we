from firebase import firebase

firebase = firebase.FirebaseApplication('https://enet5-7f9f6.firebaseio.com/', None)
data =  { 'Name': 'BANYAPON',
          'Surname': 'POOLSAWAS',
          'Gender': 'M'
          }
result = firebase.post('/person',data)
print(result)