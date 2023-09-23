import requests
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


img = Image.open('images/carlp.jpg')
I1 = ImageDraw.Draw(img)

url = "https://api.aiforthai.in.th/lpr-v2"
payload = {'crop': '1', 'rotate': '1'}
files = {'image':open('images/carlp.jpg', 'rb')}
 
headers = {
    'Apikey': "n0Ju2NMISMrBk9F1jCaNAPuZXdxWfVOi",
    }
 
response = requests.post( url, files=files, data = payload, headers=headers)
print(response.json())
data = response.json()

#ต้องใช้ font ภาษาไทย
customFont = ImageFont.truetype('fonts/PromptBold.ttf', 65)
 
# Add Text to an image
I1.text((10, 10), ""+data[0]['lpr'], font=customFont, fill =(0, 255, 0))

#I1.text((28, 36), data[0]['lpr'], fill=(0, 255, 0))
img.show()

#เซฟรูป
img.save("images/car_result.png")
