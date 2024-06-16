import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import os
import numpy as np
import math
from sklearn.cluster import DBSCAN, KMeans
from collections import defaultdict, Counter
from sklearn.metrics import pairwise_distances
import re
import matplotlib.pyplot as plt

from utils.detection_util import calculate_center_bbox, extract_text_from_image, connect_value_ocr

def calculate_threshold(width, height):
    return min(width, height) * 0.05

def process_image(image):
    
    width, height = image.shape[1], image.shape[0]

    threshold = calculate_threshold(width, height)

    results = extract_text_from_image(image)
    if results:
        results = calculate_center_bbox(results)
    return results, width, height, threshold

def cluster_labels(labels, eps, axis):
    coords = np.array([label[0] for label in labels])
    clustering = DBSCAN(eps=eps, min_samples=1, metric=lambda a, b: abs(a[axis] - b[axis])).fit(coords)

    clustered_labels = defaultdict(list)
    for label, cluster in zip(labels, clustering.labels_):
        clustered_labels[cluster].append(label)

    return clustered_labels

def combine_clustered_labels(clusters):
    combined_labels = []
    for cluster in clusters.values():
        combined_text = ' '.join([label[1] for label in cluster])
        bounding_boxes = [label[2] for label in cluster]
        centers = [label[0] for label in cluster]
        avg_center = np.mean(centers, axis=0).tolist()
        combined_labels.append((combined_text, bounding_boxes, avg_center, cluster))
    return combined_labels

def group_labels(results, width, height):
    coords = np.array([item['center_bbox'] for item in results])
    texts = [item['text'] for item in results]
    bounding_boxes = [item['bounding_box'] for item in results]

    clustering = DBSCAN(eps=15, min_samples=1, metric='euclidean').fit(coords)
    
    clustered_labels = defaultdict(list)
    for idx, label in enumerate(clustering.labels_):
        clustered_labels[label].append((coords[idx], texts[idx], bounding_boxes[idx]))

    combined_labels = []
    for cluster in clustered_labels.values():
        combined_text = ' '.join([label[1] for label in cluster])
        centers = [label[0] for label in cluster]
        avg_center = np.mean(centers, axis=0).tolist()
        combined_labels.append({
            'text': combined_text,
            'center_bbox': avg_center,
            'bounding_boxes': [label[2] for label in cluster]
        })

    threshold = calculate_threshold(width, height)
    y_labels = [(item['center_bbox'], item['text'], item['bounding_boxes']) for item in combined_labels]
    x_labels = [(item['center_bbox'], item['text'], item['bounding_boxes']) for item in combined_labels]

    y_clusters = cluster_labels(y_labels, threshold, axis=0)
    grouped_y_labels = combine_clustered_labels(y_clusters)

    x_clusters = cluster_labels(x_labels, threshold, axis=1)
    
    grouped_x_labels = combine_clustered_labels(x_clusters)

    return grouped_y_labels, grouped_x_labels

def print_grouped_labels(grouped_y_labels, grouped_x_labels, data_num):
    grouped_data_num = []
    for i in grouped_x_labels:
        grouped_data_num.append(len(i[3]))
        
    _, index = np.where([np.abs(np.array(grouped_data_num) - data_num).min() == np.abs(np.array(grouped_data_num) - data_num)])
    
    grouped_x = []
    for i in index:
        grouped_x.append(grouped_x_labels[i])
    
    grouped_x.sort(key=lambda x: x[2][1])
    grouped_y_labels.sort(key=lambda x: x[2][0])

    x_axis_label = grouped_x[-1]
    y_axis_label = grouped_y_labels[0]
    return x_axis_label, y_axis_label

def process_labels(labels):
    lst = []
    for label in labels[3]:
        lst.append(label[1])

    def remove_special_characters(text):
        return re.sub(r'[^가-힣0-9a-zA-Z\s]', '', text)

    lst = [remove_special_characters(item) for item in lst]
    return lst

def find_common_suffixes(labels):
    suffixes = []
    for item in labels:
        if isinstance(item, tuple):
            for i in item:
                item_lst = list(i)
                for j in item_lst:
                    if not j.isdigit():
                        suffixes.append(j)
        else:
            for i in item:
                item_lst = list(i)
                for j in item_lst:
                    if not j.isdigit():
                        suffixes.append(j)

    element_counts = Counter(suffixes)
    common_suffixes = [element for element, count in element_counts.items() if count >= 2]

    suffixes_dict = {}
    for common_suffix in common_suffixes:
        suffixes_dict[common_suffix] = []

    for index, common_suffix in enumerate(common_suffixes):
        for item in labels:
            item_lst = list(item)
            suffixes_status = False
            for i in item_lst:
                if i == common_suffix:
                    suffixes_status = True
            if suffixes_status:
                if ' ' in item:
                    separated_item = item.split()
                    try:
                        suffixes_dict[common_suffix].append(separated_item[index])
                    except:
                        pass
                else:
                    suffixes_dict[common_suffix].append(item)
            else:
                suffixes_dict[common_suffix].append(None)

    return common_suffixes, suffixes_dict

def update_suffixes_dict(suffixes_dict):
    for suffix in suffixes_dict:
        for index, value in enumerate(suffixes_dict[suffix]):
            if value is None:
                if suffix == '년':
                    suffixes_dict[suffix][index] = suffixes_dict[suffix][index - 1]
                elif suffix == '월':
                    if suffixes_dict[suffix][index - 1] == '12월':
                        suffixes_dict[suffix][index] = '1월'
                    else:
                        p_num = re.findall(r'\d+', suffixes_dict[suffix][index - 1])
                        if p_num:
                            previous_number = int(p_num)
                            suffixes_dict[suffix][index] = f"{previous_number + 1}월"
    return suffixes_dict

def process_keypoint_data(data):
    coordinates = [(data[i], data[i + 1]) for i in range(0, len(data)-2, 2)]
    coordinates = sorted(coordinates, key=lambda coord: coord[0])
    return coordinates

def get_image_path(filename):
    current_working_directory = os.getcwd()
    data_path = os.path.join(current_working_directory, 'data')
    file_path = os.path.join(data_path, filename)
    return file_path

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def is_real_existed(results, data):
    def find_closest_num(results, data):
        threshold = 50
        output = []
        min_dist = None
        text = None
        for i in data:
            for j in results:
                dist = euclidean_distance(i, j['center_bbox'])
                if min_dist is None:
                    if dist <= threshold:
                        min_dist = dist
                        text = j['text']
                elif isinstance(min_dist, (int, float)):
                    if dist < min_dist:
                        min_dist = dist
                        text = j['text']

            if min_dist is None:
                text = None
            min_dist = None
            output.append((i, text))

        return output

    results = find_closest_num(results, data)
    count = 0
    for result in results:
        if result[1] is None:
            count += 1
    if count >= len(results):
        return False, results
    else:
        return True, results

def text_to_number(text):
    if text is None:
        return None

    units = {
        '십': 10,
        '백': 100,
        '천': 1000,
        '만': 10000,
        '억': 100000000,
        '조': 1000000000000,
        '%' : 1,
        '$' : 1,
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
    last_part = re.findall(r'[0-9]+',last_part)
    if last_part:
        result += int(last_part[0])

    return result

def is_text_number(text):
    if text is None:
        return 3

    text = text.replace(' ', '')
    valid_korean_chars = set("일이삼사오육칠팔구십백천만억0123456789원")

    if all(char.isdigit() for char in text):
        return 0

    if all(char.isdigit() or char == ',' for char in text):
        return 1

    if all(char in valid_korean_chars for char in text):
        return 2

    return 3

def find_real_values(results, real_existence):
    new_results = []
    if real_existence:
        for result in results:
            coordinates, text = result
            if is_text_number(text) == 0:
                number = float(text)
            elif is_text_number(text) == 1:
                number = float(text.replace(',', ''))
            elif is_text_number(text) == 2:
                number = text_to_number(text)
            else:
                number = None
            new_results.append((coordinates, text, number))
    else:
        for result in results:
            new_results.append((result[0], result[1], None))

    return new_results

def is_axis_distorted(tics, locs, width, height):
    result = [0, 0, 0]
    
    if tics[0] != 0:
        result[0] = 1

    diffs = [abs(tics[i] - tics[i-1] - (tics[i-1] - tics[i-2])) for i in range(2, len(tics))]
    if any(diffs):
        result[1] = 1

    th = calculate_threshold(width, height)
    diffsLoc = [abs(locs[i][1] - locs[i-1][1] - (locs[i-1][1] - locs[i-2][1])) for i in range(2, len(tics))]
    if any(diff > (th * 2) for diff in diffsLoc):
        result[2] = 1

    if any(result):
        return True
    return False

def is_line_distorted(results, y_coord, width, height, y_labels):
    threshold = calculate_threshold(width, height)
    
    base_value = y_labels[3][0][0][1]  
    base = y_labels[3][0]  
    for result in results:
        if result[0][1] < y_coord and result[2] is not None:
            base_value = result[0][1]
            base = result
    base_len = y_coord - base_value
    base_real = base[2]
    if isinstance(base_real, (tuple, list)):
        base_real = base_real[0] if isinstance(base_real[0], (int, float)) else 1  # 첫 번째 요소가 숫자인지 확인
    base_ratio = base_len / base_real if base_real != 0 else None

    for result in results:
        if result[2] is not None:
            point_len = y_coord - result[0][1]
            point_real = result[2]
            if isinstance(point_real, (tuple, list)):
                point_real = point_real[0] if isinstance(point_real[0], (int, float)) else 1  # 첫 번째 요소가 숫자인지 확인
            point_ratio = point_len / point_real if point_real != 0 else None
            if base_ratio is not None and point_ratio is not None:
                error = abs(point_ratio - base_ratio) / base_ratio if base_ratio != 0 else 0
                if error >= threshold:
                    return True
    return False

def is_distorted(y_labels, tics, locs, results, y_coord, width, height):
    if is_y_label_exists(y_labels):
        if is_axis_distorted(tics, locs, width, height):
            return True
    else:
        if is_line_distorted(results, y_coord, width, height, y_labels):
            return True
    return False

def is_y_label_exists(y_labels):
    count = 0
    for item in y_labels[3]:
        if isinstance(item[1], (int, float)):
            count += 1
        if count >= 3:
            return True
    return False

def analyze_image(image, coordinates, box_size_ratio=0.05, n_clusters=5, color_threshold=10):
    
    image_height, image_width, _ = image.shape
    box_size = int(min(image_height, image_width) * box_size_ratio)

    # coordinates_tuples = [(coordinates[i], coordinates[i + 1]) for i in range(0, len(coordinates), 2)]
    coordinates_tuples = coordinates
    coordinates_tuples.sort(key=lambda x: x[0])  # Sort by x-coordinate
    def midpoint(x1, y1, x2, y2):
        return (x1 + x2) / 2, (y1 + y2) / 2

    def create_box(x, y, box_size):
        half_size = box_size // 2
        x_min = max(0, int(x) - half_size)
        x_max = min(image_width, int(x) + half_size + 1)
        y_min = max(0, int(y) - half_size)
        y_max = min(image_height, int(y) + half_size + 1)
        return x_min, x_max, y_min, y_max

    def cluster_colors(colors, n_clusters=5):
        kmeans = KMeans(n_clusters=min(n_clusters, len(colors)), random_state=0).fit(colors)
        color_counts = Counter(kmeans.labels_)
        clustered_colors = kmeans.cluster_centers_.astype(int)

        sorted_colors = sorted(color_counts.items(), key=lambda x: -x[1])
        dominant_color = tuple(clustered_colors[sorted_colors[0][0]])

        non_background_colors = [tuple(clustered_colors[i]) for i, count in sorted_colors if pairwise_distances([clustered_colors[i]], [dominant_color]).min() > color_threshold]

        if non_background_colors:
            second_dominant_color = non_background_colors[0]
        else:
            second_dominant_color = None

        return dominant_color, second_dominant_color

    def find_similar_color(color, color_list, threshold):
        for c in color_list:
            if np.linalg.norm(np.array(color) - np.array(c)) < threshold:
                return c
        return color

    results = []
    second_dominant_colors = []

    for x, y in coordinates:
        x_min, x_max, y_min, y_max = create_box(x, y, box_size)

        box_colors = image[y_min:y_max, x_min:x_max].reshape(-1, 3)

        dominant, second_dominant = cluster_colors(box_colors, n_clusters)

        if dominant is not None and second_dominant is not None:
            results.append((x, y, dominant, second_dominant))
            if second_dominant is not None:
                second_dominant_colors.append(second_dominant)

    normalized_second_dominant_colors = [tuple(np.round(c).astype(int)) for c in second_dominant_colors if c is not None]

    if normalized_second_dominant_colors:
        color_array = np.array(normalized_second_dominant_colors)
        kmeans = KMeans(n_clusters=min(5, len(color_array)), random_state=0).fit(color_array)
        clustered_colors = kmeans.cluster_centers_.astype(int)
        normalized_second_dominant_colors = [tuple(find_similar_color(color, clustered_colors, color_threshold)) for color in normalized_second_dominant_colors]

    if normalized_second_dominant_colors:
        second_dominant_counter = Counter(normalized_second_dominant_colors)
        most_common_second_dominant = second_dominant_counter.most_common(1)[0]
        most_common_second_dominant_color = most_common_second_dominant[0]
        print(f"Most Common Second Dominant Color: {most_common_second_dominant_color}")
    else:
        most_common_second_dominant_color = None
        print("No second dominant color found.")

    # plt.imshow(image)
    # for x, y, _, _ in results:
    #     x_min, x_max, y_min, y_max = create_box(x, y, box_size)
    #     plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='red', facecolor='none'))
    #     plt.scatter(x, y, c='blue', s=10)
    # plt.axis('off')
    # plt.show()

    return most_common_second_dominant_color

def combine_results(x_labels, new_output, second_dominant_color):
    final_results = []
    for i in range(len(x_labels[3])):
        x_label = x_labels[3][i]
        real_value = new_output[i] if i < len(new_output) else None
        final_results.append((x_label, real_value, second_dominant_color))
    return final_results

def save_grouped_results_to_file(grouped_results, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for x_label, real_value, second_dominant_color in grouped_results:
            file.write(f"x_label: {x_label[1]}, real_value: {real_value[2]}, second_dominant_color: {second_dominant_color}\n")


def data_check(image, data):
    x_thres = image.shape[1] * 0.05
    remove_index = []
    for n in range(len(data)-1):
        if data[n+1][0] - data[n][0] < x_thres:
            remove_index.append(n)
    for n, i in enumerate(remove_index):
        del data[i - n]
        
    return data
        
# Main processing
def line_detection(image,data):
    
    data = process_keypoint_data(data[2:])  # Example data; replace with actual data
    data = data_check(image, data)
    data_num = len(data)
    
    results, width, height, threshold = process_image(image)

    if not results:
        print("No results found in the image processing.")
    else:
        grouped_y, grouped_x = group_labels(results, width, height)
        x_labels, y_labels = print_grouped_labels(grouped_y, grouped_x, data_num)

        x_label_texts = process_labels(x_labels)
        
        common_suffixes, suffixes_dict = find_common_suffixes(x_label_texts)
        
        updated_dict = update_suffixes_dict(suffixes_dict) if '년' in suffixes_dict and '월' in suffixes_dict else {}

        combined_labels = []
        
        if updated_dict:
            for i in range(len(updated_dict['년'])):
                year = updated_dict['년'][i]
                month = updated_dict['월'][i]
                combined_labels.append(f"{year} {month}")

            x_labels = (x_label_texts, x_labels[1], x_labels[2], x_labels[3])

            for index, label in enumerate(x_labels[3]):
                x_labels[3][index] = (label[0], combined_labels[index], label[2])
        
        results = connect_value_ocr(results)
        results = calculate_center_bbox(results)
        
        real_existence, output = is_real_existed(results, data)
        new_output = find_real_values(output, real_existence)

        converted_y_labels = []
        for item in y_labels[3]:
            converted_text = text_to_number(item[1])
            converted_y_labels.append((item[0], converted_text, item[2]))

        y_labels = (y_labels[0], y_labels[1], y_labels[2], converted_y_labels)

        tics = [item[1] for item in y_labels[3]]
        locs = [item[0] for item in y_labels[3]]

        y_coord = x_labels[3][0][0][1]
        distorted = is_distorted(y_labels, tics, locs, new_output, y_coord, width, height)

        second_dominant_color = analyze_image(image, data, box_size_ratio=0.05, n_clusters=5)
        final_results = combine_results(x_labels, new_output, second_dominant_color)

        grouped_results = []
        for x_label, real_value, _ in final_results:
            if real_value:
                grouped_results.append((x_label, real_value, second_dominant_color))
        
        for item in grouped_results:
            print(f"x_label: {item[0][1]}, real_value: {item[1][2]}, second_dominant_color: {item[2]}")

        # final_results_path = "final_results.txt"
        # save_grouped_results_to_file(grouped_results, final_results_path)

        # print(f"Final results saved to {final_results_path}")
        if distorted:
            print("Graph Distorted:", distorted)
        else:
            print("Graph is not distorted. Skipping final results output.")
        
        result_dict_list = []

        for i in grouped_results:
            
            result_dict_list.append({'labels' : i[0][1],
                                'value' : i[1][2],
                                'value_text' : i[1][1],
                                'color' : np.array(i[2])/255.})
        return distorted, result_dict_list
