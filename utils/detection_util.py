# 작성한 코드
import io
import time
import numpy as np
import cv2
import re
import copy
import pandas as pd

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

from main import EasyPororoOcr
ocr = EasyPororoOcr()


# SUBSCRIPTION_KEY = "f9d2a2a80dfd4e74b4f807ec26aad89b"
# ENDPOINT_URL = "https://yejikim.cognitiveservices.azure.com/"
# computervision_client = ComputerVisionClient(ENDPOINT_URL, CognitiveServicesCredentials(SUBSCRIPTION_KEY))
SUBSCRIPTION_KEY = ""
ENDPOINT_URL = ""


# 3. 매칭 - 데이터 & OCR
# 3-1. OCR 읽기

def extract_text_from_image(img):
    
    if SUBSCRIPTION_KEY and ENDPOINT_URL:
        # 이미지 데이터를 읽어들임
        img_byte = cv2.imencode('.png', img)[1].tobytes()
        sbuf = io.BytesIO(img_byte)

        # OCR 요청 보내기
        response = computervision_client.read_in_stream(sbuf, raw=True)
        operation_location = response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        # OCR 결과 기다리기
        while True:
            read_result = computervision_client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            time.sleep(1)

        # OCR 결과 처리
        results = []
        if read_result.status == OperationStatusCodes.succeeded:
            # roi_img = img.copy()

            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    text = line.text
                    box = list(map(int, line.bounding_box))
                    (tlX, tlY, trX, trY, brX, brY, blX, blY) = box
                    pts = ((tlX, tlY), (trX, trY), (brX, brY), (blX, blY))
                    text = re.sub('[,;.]','',text)

                    result_dict = {
                        "text": text,
                        "bounding_box": pts
                    }
                    results.append(result_dict)
        
        return results
    
    else:
        text = ocr.run_ocr(img, debug=False)
        results = []
        for bbox, txt in text:
            txt = re.sub('[,;.]','',txt)
            results.append({'text': txt,
                            'bounding_box':bbox})
        return results


# 3-2. OCR 결과값 후처리 
# return: results that includes center point of the bbox 
#         {results, 'center_bbox':(x,y)}
def calculate_center_bbox(results):
    center_bbox_key = 'center_bbox'
    
    results = copy.deepcopy(results)
    
    # each bounding box에서 center을 계산 후 저장 
    for result in results:
        bounding_box = result['bounding_box']
        # 네 개의 좌표의 x와 y 값을 모두 더한 후에 2로 나누어서 중앙점을 계산
        center_x = (bounding_box[0][0] + bounding_box[1][0] + bounding_box[2][0] + bounding_box[3][0]) / 4
        center_y = (bounding_box[0][1] + bounding_box[1][1] + bounding_box[2][1] + bounding_box[3][1]) / 4
        center = (center_x, center_y)
        result[center_bbox_key] = center

    return results

def connect_value_ocr(matching_result):
    temp_result = calculate_center_bbox(matching_result)
    df = pd.DataFrame(temp_result)

    x_thred = int(df['bounding_box'].apply(lambda x: (x[1][0] + x[2][0] - x[0][0] - x[3][0])/2).median()* 0.8)
    y_thred = int(df['bounding_box'].apply(lambda x: (x[2][1] + x[3][1] - x[0][1] - x[1][1])/2).median()* 1.25)

    temp = [[] for _ in df.index]
    for target_index, target_value in df['center_bbox'].items():
        for check_index, check_value in df['center_bbox'].items():
            if check_index == target_index:
                continue
            elif abs(target_value[0]-check_value[0]) < x_thred and abs(target_value[1]-check_value[1]) < y_thred:
                temp[target_index].append(check_index)

    matched_list = []
    check_temp_list = []
    for n,i in enumerate(temp):
        if i:
            if n not in check_temp_list or i[0] not in check_temp_list:
                matched_list.append((n,i[0]))
            check_temp_list.append(n)
            check_temp_list.append(i[0])
    
    if not matched_list:
        return matching_result
    
    new_dict_list = []

    for i in matched_list:
        a = df.iloc[[i[0]],:]
        b = df.iloc[[i[1]],:]
        
        x_vals = []
        y_vals = []
        for i in a['bounding_box'].iloc[0]:
            x_vals.append(i[0])
            y_vals.append(i[1])
        for i in b['bounding_box'].iloc[0]:
            x_vals.append(i[0])
            y_vals.append(i[1])
            
        min_x = min(x_vals)
        max_x = max(x_vals)
        min_y = min(y_vals)
        max_y = max(y_vals)

        new_dict_list.append({'text':a['text'].iloc[0] + b['text'].iloc[0],
                    'bounding_box':((min_x,min_y),(max_x,min_y),(max_x,max_y),(min_x,max_y))})
    remains = df[df.apply(lambda x: x.name not in check_temp_list, axis=1)]
    
    for _, i in remains.iterrows():
        new_dict_list.append({'text':i['text'],
                    'bounding_box':i['bounding_box']})
    return new_dict_list