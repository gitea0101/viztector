
import numpy as np
import re

from utils.detection_util import calculate_center_bbox, extract_text_from_image


# 1. keypoint data 후처리

# return: {pie # : [[(midx, midy), (edge1x, edge1y), (edge2x, edge2y), (centerx, centery)], ...], 
#          ...}
def process_keypoint_data(data):
    # elements round to second decimal places & make (x,y) 
    transformed_data = [
        [(round(row[0], 2), round(row[1], 2)),
         (round(row[2], 2), round(row[3], 2)),
         (round(row[4], 2), round(row[5], 2)),
         (round(row[6], 2), round(row[7], 2))] 
        for row in data]
     
    centers = []
    for i in range(len(transformed_data)):
        if transformed_data[i][3] not in centers:
            centers.append(transformed_data[i][3])
    centers = sorted(centers, key=lambda x: x[0])    

    transformed_datas = {}
    for i in range(len(transformed_data)):
        center = transformed_data[i][3]
        if centers.index(center) not in transformed_datas.keys():
            transformed_datas[centers.index(center)] = [transformed_data[i]]
        else:
            transformed_datas[centers.index(center)] = transformed_datas[centers.index(center)] + [transformed_data[i]]

    return transformed_datas


# 2. data의 각도 계산 
import math

def vector_subtract(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1])

def vector_length(v):
    return math.sqrt(v[0]**2 + v[1]**2)

def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def cross_product(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]

# (center, mid) vector가 (center, edge1) & (center, edge2) 내각에 있는지 여부 확인 
# return: True - exist 
#         False - not exist  
def is_vector_in_angle(a, b, c, d):
    ab = vector_subtract(b,a)
    ac = vector_subtract(c,a)
    ad = vector_subtract(d,a)

    cross_ab_ac = cross_product(ab, ac)
    cross_ab_ad = cross_product(ab, ad)
    
    # 부호 비교
    # 부호가 다르면 False
    if (cross_ab_ac >= 0 and cross_ab_ad >= 0) or (cross_ab_ac <= 0 and cross_ab_ad <= 0):
        return True
    return False

# 내각 구하기 
def angle_between(v1, v2):
    dot = dot_product(v1, v2)
    len1 = vector_length(v1)
    len2 = vector_length(v2)
    cos_theta = dot / (len1 * len2)
    angle = math.acos(cos_theta)
    return math.degrees(angle)

# 각도 계산
# return: a list of angles in data respectively 
def calculate_angles(data):

    angles = []
    
    for row in data:
        angle_dict = {}
        
        center = row[3]
        vec1 = vector_subtract(row[1], center)
        vec2 = vector_subtract(row[2], center)
        reference_vec = vector_subtract(row[0], center)

        # 내각 / 외각 여부 확인 후 각도 구하기 
        if is_vector_in_angle(row[3], row[1], row[2], row[0]):
            angle = round(angle_between(vec1, vec2), 2)
        else:
            angle = round(360 - angle_between(vec1, vec2), 2)
        
        angle_dict['angle'] = angle
        angle_dict['percentage'] = round(angle/360, 2)
        angle_dict['midpoint'] = row[0]
        angles.append(angle_dict)
    
    return angles   

# return: results that center_bbox in dete_results and short phrases 
def process_results(results, dete_results): 
    for index, result in enumerate(results): 
        # center_bbox x not locates in dete_results 
        center = result['center_bbox']
        if not (dete_results[0] <= center[0] <= dete_results[2]):
            results.remove(result)
        # center_bbox y not locates in dete_results 
        elif not (dete_results[1] <= center[1] <= dete_results[2]):
            results.remove(result)
        else:
            # too long text 
            text = result['text']
            if len(text.split(' ')) >= 5:
                results.remove(result)                
    return results

# return: True if metrics is percentage, False else 
def find_metrics(results):
    keywords = ['%', 'percent', 'percentage', '퍼센트', '백분율']
    for result in results:
        for keyword in keywords:
            text = result['text']
            if keyword in text:
                return True
    return False

# return : remove special characters 
#          change 'text' to 'onlyNum' if contains num
#          change 'text' to 'num' if contains string & num
def process_text(results):
    # return: True - only digit, False - digit + alpha 
    def classify_text(text):
        has_digit = any(char.isdigit() for char in text)  
        has_alpha = any(char.isalpha() for char in text)     
        if has_digit and has_alpha:  
            return False
        elif has_digit:  
            return True
            
    pattern = r'[*%+=@$#()]'
    results = [{'text': re.sub(pattern, '', result['text']), **{k: v for k, v in result.items() if k != 'text'}} for result in results]
    for result in results:
        result['text'] = re.sub(pattern, '', result['text'])
        if classify_text(result['text']):
            result['onlyNum'] = result.pop('text')
        elif classify_text(result['text']) == False:
            result['num'] = result.pop('text')
    return results 



# 3-3. OCR 결과값 & 실제값 매칭 

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# real로 뽑힌 숫자가 중복되는 경우, 그 다음 가까운 거리의 숫자 찾기 
# return: second closest num/onlyNum, second closest center_bbox 
def find_next_closest_element(point, results, used_bbox):
    min_distance = float('inf')
    closest_element = None
    closest_bbox = None
    for result in results:
        if 'num' in result or 'onlyNum' in result:
            center_bbox = result.get('center_bbox', None)
            if center_bbox:
                dist = euclidean_distance(point, center_bbox)
                if dist < min_distance:
                    num_value = None
                    if 'num' in result:
                        num_value = result['num']
                    elif 'onlyNum' in result:
                        num_value = result['onlyNum']
                    if num_value not in used_bbox:
                        min_distance = dist
                        closest_element = result
                        closest_bbox = center_bbox
    return closest_element, closest_bbox


# 중복없이 가장 가까운 숫자 찾기
# return: {..., 'value':'closest onlyNum', 'labels'}
# if num --> return {..., 'value':'closest onlyNum', 'labels': closest labels}
def find_closest_num(angles, results):
    
    # ['num'] value 숫자와 레이블 구분 
    def split_num_string(num_value):
        num_parts = [part.strip() for part in num_value.split() if part.strip()] 
        number = None
        labels = []
        for part in num_parts:
            if part.isdigit():
                number = int(part) 
            else:
                labels.append(part)
        label = ' '.join(labels)    
        return number, label 
        
    used_bbox = set()  
    
    for i in angles:
        for j, item in enumerate(angles[i]):
            midpoint = item['midpoint']
            closest_element, closest_bbox = find_next_closest_element(midpoint, results, used_bbox)  
            if closest_element:
                if 'num' in closest_element:
                    num_value = closest_element['num']
                    # 중복 없는 
                    if closest_bbox not in used_bbox:
                        angles[i][j]['value'] = split_num_string(num_value)[0]
                        angles[i][j]['real_loc'] = closest_bbox
                        angles[i][j]['labels'] = split_num_string(num_value)[1]
                        used_bbox.add(closest_bbox)
                    # 중복이라면 다음으로 가까운 값 
                    else:
                        closest_element, closest_bbox = find_next_closest_element(midpoint, results, used_bbox)
                        if closest_element:
                            angles[i][j]['value'] = split_num_string(closest_element['num'])[0]
                            angles[i][j]['real_loc'] = closest_bbox
                            angles[i][j]['labels'] = split_num_string(closest_element['num'])[1]
                            used_bbox.add(closest_bbox)
                elif 'onlyNum' in closest_element:
                    only_num_value = closest_element['onlyNum']
                    # 중복된 값이 아니라면
                    if closest_bbox not in used_bbox:
                        angles[i][j]['value'] = only_num_value
                        angles[i][j]['real_loc'] = closest_bbox
                        used_bbox.add(closest_bbox)
                    # 중복된 값이라면 다음으로 가까운 값
                    else:
                        closest_element, closest_bbox = find_next_closest_element(midpoint, results, used_bbox)
                        if closest_element:
                            angles[i][j]['value'] = closest_element['onlyNum']
                            angles[i][j]['real_loc'] = closest_bbox
                            used_bbox.add(closest_bbox)
    return angles




# 3-4. 3-3결과에서 label 매칭
# 이미지 크기에 따라 find_labels에 활용될 거리 threshold 찾기 
def find_threshold(data):
    flatten_data = [item for sublist in data.values() for item in sublist]
    max_x = max(coord[0] for sublist in flatten_data for coord in sublist)
    max_y = max(coord[1] for sublist in flatten_data for coord in sublist)
    if (max_x > 500) or (max_y > 500):
        return 1000
    else:
        return 100
        
def find_labels(angles, results, threshold):
    dists = {}
    for i in range(len(angles)):
        dists[i] = {}
        for result in results:
            if 'text' in result.keys():
                text = result['text']
                dists[i][text] = []
     
    labels_lst = []
    used_labels_lst = []
    for i in range(len(angles)):
        for jth_angle, _ in enumerate(angles[i]):
            labels_lst.append([])
            real_loc = angles[i][jth_angle]['real_loc']
            for result in results:
                if 'text' in result.keys():
                    text = result['text']
                    dist = euclidean_distance(result['center_bbox'], real_loc)
                    dists[i][text].append(dist)
                    if dist <= threshold:               
                        # 중복 값이 없으면
                        if text not in used_labels_lst:
                            labels_lst[jth_angle].append(text)
                            used_labels_lst.append(text)
                        # 중복 존재
                        else: 
                            min_value = min(dists[i][text])
                            # min_index = dists[i][text].index(min(dists[i][text]))
                            # second_min_value = min(x for x in dists[i][text] if x != min_value)
                            second_min_index = dists[i][text].index(min(x for x in dists[i][text] if x != min_value))
                            if min_value == dists:
                                labels_lst[i].pop(second_min_index)
                                labels_lst[i].append(text)

    for i in range(len(angles)):
        for j, labels in enumerate(labels_lst):         
            angles[i][j]['labels'] = ' '.join(labels)                 
        
    return angles



# 4. 왜곡 판단 
def is_distorted(angles):
    sum_value = 0
    
    for item in angles[0]:
        real = item['value']
        sum_value += float(real)
    
    real_mul = 100/sum_value

    for item in angles[0]:
        percentage = item['percentage'] * 100
        real = float(item['value']) * real_mul
        if abs(percentage - real) > 2.0:
            return True
    return False


# 5. 넘길 형태로 바꾸기

# 5-1. sort clockwise
def sort_clockwise(angles, center):
    # midpoint를 center를 기준으로 시계 방향으로
    def compare(item):
        x, y = item['midpoint']
        cx, cy = center[0]
        
        rad = -np.arctan2(cx - x, cy - y)
        if rad < 0 :
            rad = 2*np.pi + rad
        
        return rad

    sorted_angles = sorted(angles[0], key=compare)
    index = np.argsort([compare(i) for i in angles[0]])

    return sorted_angles, index


# 5-2. 색 추출 
def add_colors(angles, img):
    
    def get_pixel_color(img, x, y):  
        x_int = int(x)
        y_int = int(y)
        rgb = img[y_int,x_int,:]
        return tuple(rgb/ 255.)
        
    for angle in angles[0]:
        x, y = angle['midpoint']
        rgb = get_pixel_color(img, x, y)
        angle['color'] = rgb
    return angles

# 파이 그래프 중심점이 2번 인덱스인지 확인
def pie_relocation(data):
    
    pie_points = []

    # 키 포인트 추출
    for i in data:
        pie_points = pie_points + np.array((i[2:-2:2],i[3:-2:2])).T.tolist()
        
    pie_points = np.array(pie_points)
    
    # 부채꼴 갯수가 2개 이상일때
    if len(data) > 2:

        values, counts = np.unique(pie_points,axis=0, return_counts=True)

        point_counts = sorted(counts)
        
        # 가장많이 겹치는 점이 중심점
        if point_counts[-1] not in point_counts[:-1]:
            pie_midle = values[np.argmax(counts)]
            pie_points = pie_points.reshape(-1,3,2)
            
            # 중심점이 2번 인덱스인지 확인
            for pp in pie_points:
                is_midle = pp == pie_midle
                pm_loc = np.argmax(is_midle.sum(axis=1))

                if pm_loc != 2:
                    temp = pp[pm_loc].copy()
                    pp[pm_loc] = pp[2]
                    pp[2] = temp
    # 부채꼴 갯수가 2개 일때
    else:
        
        circular_mid = []
        
        for i in data:
            circular_mid = circular_mid + np.array((i[0:1],i[1:2])).T.tolist()
        
        circular_mid = np.array(circular_mid)
        
        check = []
        for cm in circular_mid[:1]:
            
            ohter_cm = circular_mid.tolist()
            ohter_cm.remove(list(cm))
            
            # 두 부채꼴 중심점과 키 포인트 사이의 방향이 가장 다른 점이 중심점
            for ocm in ohter_cm:
                for pp in pie_points[:3]:
                    check.append(np.dot((pp[0]-cm[0], pp[1]-cm[1]), (pp[0]-ocm[0], pp[1]-ocm[1])) \
                        / (np.linalg.norm((pp[0]-cm[0], pp[1]-cm[1])) * np.linalg.norm((pp[0]-ocm[0], pp[1]-ocm[1]))))
            
        pie_midle = pie_points[:3][np.argmin(check)].copy()
        
        pie_points = pie_points.reshape(-1,3,2)

        # 중심점이 2번 인덱스인지 확인
        for pp in pie_points:
            is_midle = pp == pie_midle
            pm_loc = np.argmax(is_midle.sum(axis=1))

            if pm_loc != 2:
                temp = pp[pm_loc].copy()
                pp[pm_loc] = pp[2]
                pp[2] = temp
                
    pie_points = pie_points.reshape(-1,6)
    
    for i in range(len(data)):
        for j in range(2,8):
            data[i][j] = pie_points[i][j-2]
    
    return data

def cal_start_point(data, index):
    i = data[0][index[0]]
    
    rad1 = abs(np.arctan2(i[3][0] - i[1][0], i[3][1] - i[1][1]))
    rad2 = abs(np.arctan2(i[3][0] - i[2][0], i[3][1] - i[2][1]))
        
    if rad1 < rad2:
        start_p = (i[1][0],i[1][1])
    else:
        start_p = (i[2][0],i[2][1])
        
    pie_mid = (i[3][0],i[3][1])
    
    return start_p, pie_mid

def pie_detection(img, data):
    # 테스트
    # 1. 모델 데이터 후처리
    data = pie_relocation(data)
    data = process_keypoint_data(data)
    
    center = [data[i][0][3] for i in range(len(data))]

    # 2. 각도 계산 
    angles = {}
    for index in data:
        angles[index] = calculate_angles(data[index])

    # 3. 매칭 
    # 3-1. OCR 읽기
    results = extract_text_from_image(img)
    # 단위 찾기 
    metrics = find_metrics(results)
    print(metrics)
    # 3-2. OCR 후처리 
    results = calculate_center_bbox(results)
    # dete_results = [11.8592091, 30.3664589, 679.894287, 507.981628]
    # results = process_results(results, dete_results)
    results = process_text(results)
    # 3-3. 실제 & OCR 값 매칭 
    angles = find_closest_num(angles, results)
    # 3-4. 3-3 결과에서 label 매칭 
    angles = find_labels(angles, results, 100)

    # 4. 왜곡 탐지 
    
    is_dist = is_distorted(angles)
    print(is_dist)

    # 5. 교정본 raw data 보내기 
    angles = add_colors(angles, img)
    angles, sort_index = sort_clockwise(angles, center)

    # 그래프 시작점 계산
    start_point, pie_mid = cal_start_point(data,sort_index)
    
    # 시작 부채꼴 중앙과 그래프 중앙사이의 각도
    start_angle = np.arctan2(
    -angles[0]['midpoint'][1] + data[0][0][3][1],
    angles[0]['midpoint'][0] - data[0][0][3][0])
    
    # 그래프 중앙점 과 부채꼴 각도의 1/2 덧샘
    start_angle = np.rad2deg(start_angle) + angles[0]['percentage']/2 * 360
    
    return is_dist, angles, start_angle, sort_index, start_point, pie_mid


