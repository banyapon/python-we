import face_recognition
from PIL import Image

image = face_recognition.load_image_file('team.jpg')
face_locations = face_recognition.face_locations(image)
for f in face_locations:
    top, right, bottom, left = f
    face_img = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_img)
    pil_image.show()