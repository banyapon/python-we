import face_recognition

#dataset
database_image = face_recognition.load_image_file("3d.jpg")
data_base_face_encoding = face_recognition.face_encodings(database_image)[0]
#compare
sample_image = face_recognition.load_image_file("3dmale.jpg")
sample_face_encoding = face_recognition.face_encodings(sample_image)[0]

results = face_recognition.compare_faces([data_base_face_encoding],
sample_face_encoding)
if results[0] == True:
    print("It's me Banyapon")
else:
    print("I don't know him!")