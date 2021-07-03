from logging import debug
from flask import Flask, render_template, request
import os
import numpy as np
import cv2
import pytesseract as pyt
from PIL import Image
import requests as rq
import xmltodict as xml
import json as js

def myfunction(frame):
    car_cascade = cv2.CascadeClassifier('indian_license_plate.xml')
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 50)
        image_filter = cv2.bilateralFilter(gray, 11, 17, 17)
        edge = cv2.Canny(image_filter, 30, 200)
        for (x,y,w,h) in cars:
            location = np.array([[[x+w,y]],[[x,y]],[[x,y+h]],[[x+w,y+h]]], dtype=np.int32)
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = cv2.bitwise_and(frame, frame, mask=mask)
    else:
        text = "No Plate recognized"
        return text
    (x,y) = np.where(mask==255)
    (x1,y1) = (np.min(x), np.min(y))
    (x2,y2) = (np.max(x), np.max(y))

    crpped_image = gray[x1:x2+1, y1:y2+1]
    pyt.pytesseract.tesseract_cmd = r'C:\Users\Asus\AppData\Local\Programs\Tesseract-OCR\tesseract'
    text = pyt.image_to_string(crpped_image, lang='eng', config='--psm 6')

    x=0
    number = ''
    num = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    for i in text:
        for j in num:
            if i == j:
                number += i
    
    info =  vehicle_information(number)
    return number, info

def vehicle_information(number):
    request = rq.get(f"http://www.regcheck.org.uk/api/reg.asmx/CheckIndia?RegistrationNumber={number}&username=pokjm")
    num_1 = xml.parse(request.content)
    num_2 = js.dumps(num_1)
    num_3 = js.loads(num_2)
    num_4 = js.loads(num_3['Vehicle']['vehicleJson'])
    return num_4


app=Flask(__name__)
root=os.path.dirname(os.path.abspath(__file__))
fname=None
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=["POST"])
def upload():
    global fname
    path=os.path.join(root, 'static/')
    if not os.path.isdir(path):
        os.mkdir(path)
   
    for file in request.files.getlist("user_img"):
        f_name=file.filename
        dst="/".join([path, f_name])
        fname = f_name
        file.save(dst)
        return render_template('index.html', data=f_name)

@app.route('/info')
def get_info():
    number, a=myfunction(cv2.imread(f"static/{fname}"))
    number = number
    Owner = a['Owner']
    Description = a['ModelDescription']
    Model = Description['CurrentTextValue']
    RegistrationDate = a['RegistrationDate']
    Insurance = a['Insurance']
    Location = a['Location']
    VechileIdentificationNumber = a['VechileIdentificationNumber']
    EngineNumber = a['EngineNumber']
    FuelType = a['FuelType']
    FuelType = FuelType["CurrentTextValue"]
    Fitness = a['Fitness']
    st=f'''Owner ====>{Owner}+Vehicle Number ===> {number}+Model ====>{Model}+Registration date ====>{RegistrationDate}+Engine Number====>
    {EngineNumber}+Vehicle Identification Number ====>{VechileIdentificationNumber}+Insurance ====>{Insurance}+Fuel Type ====>{FuelType}+Fitness ====>{Fitness}+Location ====>{Location}'''
    return st

app.run(debug=True, port=5001)