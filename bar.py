# 만든 코드
import re
import numpy as np
import pandas as pd
from utils.detection_util import extract_text_from_image, connect_value_ocr

def extract_numeric_value(text):
    units = {
        '십': 10,
        '백': 100,
        '천': 1000,
        '만': 10000,
        '억': 100000000,
        '조': 1000000000000,
        '%' : 1,
        '$' : 1
    }

    num_dict = {
        '일': 1,
        '이': 2,
        '삼': 3,
        '사': 4,
        '오': 5,
        '육': 6,
        '칠': 7,
        '팔': 8,
        '구': 9
    }

    for k, v in num_dict.items():
        text = text.replace(k, str(v))

    text = text.replace('원', '').replace(' ', '')

    pattern = re.compile(r'([0-9]*)([조억만천백십%$])')
    matches = pattern.findall(text)

    result = 0
    current_unit_value = 1
    for value, unit in matches[::-1]:
        if not value:
            value = 1
        result += int(value) * units[unit]
        current_unit_value = units[unit]

    last_part = pattern.sub('', text)
    if last_part:
        try:
            result += int(last_part)
        except:
            return None

    return result

# 좌상단 좌표 추출 함수
def extract_top_left(bbox):
    return bbox[0]

# 거리 계산 함수
def calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

# 색상 추출 함수
def get_pixel_color(img, x, y):
    x_int = int(x)
    y_int = int(y)
    rgb = img[y_int,x_int,:]
    return tuple(rgb/ 255.)


def bar_detection(image,keypoint_data):
    ocr_data = extract_text_from_image(image)

    # 숫자로 변환할 수 있는 텍스트 추출
    numeric_ocr_data = [entry for entry in ocr_data if extract_numeric_value(entry['text']) is not None]

    # 숫자 값들을 리스트로 추출
    ocr_values = [extract_numeric_value(entry['text']) for entry in numeric_ocr_data]

    filtered_ocr_data = [entry for entry in numeric_ocr_data if abs(extract_numeric_value(entry['text']))]
    
    filtered_ocr_data = connect_value_ocr(filtered_ocr_data)
    
    # 필터링된 OCR 값 리스트
    filtered_ocr_values = [extract_numeric_value(entry['text']) for entry in filtered_ocr_data]

    # keypoint_data에서 바의 좌표 추출 및 정렬
    keypoint_coor = []
    last_part = keypoint_data

    for item in last_part:
        top_left = [item[4], item[5]]
        bottom_right = [item[2], item[3]]
        keypoint_coor.append(top_left)
        keypoint_coor.append(bottom_right)

    # x좌표를 기준으로 정렬
    keypoint_coor.sort(key=lambda x: x[0])

    # 좌상단과 우하단 좌표로 묶기
    bars = []
    for i in range(0, len(keypoint_coor), 2):
        if i + 1 < len(keypoint_coor):
            top_left = keypoint_coor[i]
            bottom_right = keypoint_coor[i + 1]
            bars.append([top_left, bottom_right])

    # 바 네이밍
    named_bars = {f"bar{i+1}": bar for i, bar in enumerate(bars)}

    # 매칭 결과 저장
    matching_result = []

    # 바의 가까운 OCR 데이터에 대해 가장좌표를 찾고 매칭
    for bar_name, bar in named_bars.items():
        bar_top_left = bar[0]
        min_distance = float('inf')
        closest_bar = None
        
        for ocr in filtered_ocr_data:
            ocr_top_left = extract_top_left(ocr["bounding_box"])
            
            distance = calculate_distance(ocr_top_left, bar_top_left)
            
            if distance < min_distance:
                min_distance = distance
                closest_bar = bar_name
                ocr_text = ocr["text"]
                
        bar_botterm_left = (bar[0][0],bar[1][1])
        
        min_distance2 = float('inf')
        ocr_text2 = closest_bar
        for ocr2 in ocr_data:
            ocr_top_left = extract_top_left(ocr2["bounding_box"])
            if ocr_top_left[1] < image.shape[0] * 0.7:
                continue
            else:
                distance2 = calculate_distance(ocr_top_left, bar_botterm_left)
                if distance2 < min_distance2:
                    min_distance2 = distance2
                    ocr_text2 = ocr2["text"]
        
        matching_result.append({
            "bar": closest_bar,
            "value_text": ocr_text,
            "value": extract_numeric_value(ocr_text),
            "bar_coordinates": named_bars[closest_bar],
            "labels": ocr_text2
        })

    # 바 이름 순서대로 정렬
    matching_result.sort(key=lambda x: int(x["bar"][3:]))

    # 가장 긴 바를 기준으로 설정
    base_bar = max(matching_result, key=lambda x: abs(x['bar_coordinates'][1][1] - x['bar_coordinates'][0][1]))
    base_bar_length = abs(base_bar['bar_coordinates'][1][1] - base_bar['bar_coordinates'][0][1])
    base_ocr_value = base_bar['value']
    base_ratio = base_ocr_value / base_bar_length

    # 결과 출력 및 왜곡 판정
    threshold = 0.05  # 5%의 오차 범위
    overall_distortion = False

    for result in matching_result:
        bar_name = result['bar']
        ocr_value = result['value']
        bar_coordinates = result['bar_coordinates']
        bar_length = abs(bar_coordinates[1][1] - bar_coordinates[0][1])
        
        # 비율 계산
        ratio = ocr_value / bar_length if bar_length != 0 else 0
        
        # 오차 범위 계산
        error = abs(ratio - base_ratio) / base_ratio if base_ratio != 0 else 0
        
        # 왜곡 판정
        distortion = "왜곡 아님" if error <= threshold else "왜곡"
        if distortion == "왜곡":
            overall_distortion = True
        

        print(f"Bar: {bar_name}, OCR Value: {ocr_value}, Coordinates: {bar_coordinates}, Bar Length: {bar_length}, Distortion: {distortion}")

    # 전체 그래프 왜곡 판정 결과
    print("그래프 왜곡:", "왜곡 아님" if not overall_distortion else "왜곡")

    # 이미지에서 색상 추출 및 추가
    for result in matching_result:
        bar_coordinates = result['bar_coordinates']
        midpoint_x = (bar_coordinates[0][0] + bar_coordinates[1][0]) / 2
        midpoint_y = (bar_coordinates[0][1] + bar_coordinates[1][1]) / 2
        bar_color = get_pixel_color(image, midpoint_x, midpoint_y)
        result['color'] = bar_color

    # 결과 출력
    if overall_distortion:
        print("왜곡")
        # print("Filtered OCR Values:", filtered_ocr_values_sorted)
        for i, result in enumerate(matching_result):
            bar_name = f"bar{i+1}"
            ocr_value = result['value']
            bar_color = result['color']
            # print(f"Bar: {bar_name}, OCR Value: {ocr_value}, Color: {bar_color}, Coordinates: {result['bar_coordinates']}")

            # 수정된 좌상 좌표 계산
            new_bar_length = ocr_value / base_ratio
            new_top_left_y = round(result['bar_coordinates'][1][1] - new_bar_length, 2)
            
            # 업데이트된 바 좌표
            new_bar_coordinate = [[round(result['bar_coordinates'][0][0], 2), new_top_left_y], [round(result['bar_coordinates'][1][0], 2), round(result['bar_coordinates'][1][1], 2)]]
            result['new_bar_coordinate'] = new_bar_coordinate
            # print(f"Updated Coordinates: {new_bar_coordinate}")

        # 연도별 OCR 값 출력
        # print("Year and Value Pairs:")
        # print(year_value_pairs)
    else:
        print("왜곡 아님")
        for i, result in enumerate(matching_result):
            bar_name = f"bar{i+1}"
            ocr_value = result['value']
            bar_color = result['color']
            # print(f"Bar: {bar_name}, OCR Value: {ocr_value}, Color: {bar_color}, Coordinates: {result['bar_coordinates']}")

            # 수정된 좌상 좌표 계산
            new_bar_length = ocr_value / base_ratio
            new_top_left_y = round(result['bar_coordinates'][1][1] - new_bar_length, 2)
            
            # 업데이트된 바 좌표
            new_bar_coordinate = [[round(result['bar_coordinates'][0][0], 2), new_top_left_y], [round(result['bar_coordinates'][1][0], 2), round(result['bar_coordinates'][1][1], 2)]]
            result['new_bar_coordinate'] = new_bar_coordinate
    
    return overall_distortion, matching_result
