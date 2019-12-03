import requests
import base64
import cv2
#
test_img = 'data/test.png'
#
img_encode = base64.b64encode(cv2.imencode('.png', img = cv2.imread(test_img))[1])
res = requests.post('http://127.0.0.1:5000/api/predict', json={"image": str(img_encode, 'utf-8')})
#
if res.ok
    print res.json()
