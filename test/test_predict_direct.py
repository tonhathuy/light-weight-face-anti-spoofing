import base64
import urllib.parse
import requests
import json
import timeit
import sys
import io
import cv2
import numpy as np
import time
start_time = time.time()
#url = 'http://0.0.0.0:2341/predict'
#url = 'http://service.aiclub.cs.uit.edu.vn/face_anti_spoofing/predict/'
url = 'http://service.aiclub.cs.uit.edu.vn/light_weight_face_anti_spoofing/predict'
####################################
image_path = "test_face_1.png"
####################################
img = cv2.imread(image_path)
is_success, buffer = cv2.imencode('.png', img)
f = io.BytesIO(buffer)
image_encoded = base64.encodebytes(f.getvalue()).decode('utf-8')
####################################
data ={"images": [image_encoded]}
headers = {'Content-type': 'application/json'}
data_json = json.dumps(data)
response = requests.post(url, data = data_json, headers=headers)
response = response.json()
print(response)
print('time', time.time()-start_time)

