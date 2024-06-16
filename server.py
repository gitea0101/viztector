from val_extraction import pre_load_nets, get_groups, parse_args
from requests import request
from requests.exceptions import HTTPError
from time import sleep
import re
import websockets
import asyncio
import json
import cv2
import torch
import numpy as np
import pandas as pd
from collections import Counter
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pie import pie_detection
from bar import bar_detection
from line import line_detection

ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0'

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches ='tight')
    buf.seek(0)
    img = plt.imread(buf)
    return img

def plot_pie1(img, df, start_p, pie_mid,is_dist):
    if not is_dist:
        return None

    patches = []

    fig, ax = plt.subplots(1)
    ax.set_axis_off()
    
    real_mul = 100/(df['value'].astype('float').sum())

    for n, theta in enumerate(df['value'].astype('float') * 2 * np.pi * real_mul / 100):
        
        poly_point = []
        
        poly_point.append((pie_mid[0],pie_mid[1]))
        poly_point.append((start_p[0],start_p[1]))
        
        for i in range(1,101):
            start_p = ((start_p[0] - pie_mid[0]) * np.cos(theta/ 100) - (start_p[1] - pie_mid[1]) * np.sin(theta/ 100) + pie_mid[0],
            (start_p[0]- pie_mid[0]) * np.sin(theta/ 100) + (start_p[1] - pie_mid[1]) * np.cos(theta/ 100) + pie_mid[1])
            poly_point.append(start_p)

        patches.append(Polygon(poly_point, fill=None ,edgecolor='blue',ls='solid',lw=2))

        
    for patch in patches:
        ax.add_artist(patch)


    ax.imshow(img)
    
    fig = plt.gcf()
    plot_img = fig2img(fig)
    plot_img = np.array(plot_img*255., dtype='uint8')
    
    return plot_img

def plot_pie2(df,start_angle):
    plt.figure()
    if 'labels' in df.columns:
        plt.pie(df['value'], colors=df['color'], labels=df['labels'], autopct='%.1f%%', startangle=start_angle, counterclock=False)
    else:
        plt.pie(df['value'], colors=df['color'], autopct='%.1f%%', startangle=start_angle, counterclock=False)

    fig = plt.gcf()
    plot_img = fig2img(fig)
    plot_img = np.array(plot_img*255., dtype='uint8')
    
    return plot_img

def plot_bar1(img, df, is_dist):
    if not is_dist:
        return None
    patches = []

    fig, ax = plt.subplots(1)
    ax.set_axis_off()

    for n, p in enumerate(df['new_bar_coordinate']):
        
        poly_point = []
        
        tl = (p[0][0],p[0][1])
        br = (p[1][0],p[1][1])
        
        poly_point.append(tl)
        poly_point.append((br[0],tl[1]))
        poly_point.append(br)
        poly_point.append((tl[0],br[1]))
        
        patches.append(Polygon(poly_point, fill=None ,edgecolor='red',ls='solid',lw=1))
        
        for patch in patches:
            ax.add_artist(patch)

    ax.imshow(img)
    
    fig = plt.gcf()
    plot_img = fig2img(fig)
    plot_img = np.array(plot_img*255., dtype='uint8')
    
    return plot_img
    
def plot_bar2(df):
    plt.figure()
    
    if 'labels' in df.columns:
        bar = plt.bar(df['bar'],df['value'],color=df['color'], tick_label = df['labels'], width=0.6)
        plt.ylim((0,max(df['value'])*1.1))
        plt.xticks(rotation=45)
    else: 
        bar = plt.bar(df['bar'],df['value'],color=df['color'], width=0.6)
        plt.ylim((0,max(df['value'])*1.1))
        
    for rect, value_text in zip(bar, df['value_text']):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, value_text, ha='center', va='bottom', size = 12)

    fig = plt.gcf()
    plot_img = fig2img(fig)
    plot_img = np.array(plot_img*255., dtype='uint8')
    return plot_img

def plot_line1(df, is_dist):
    return None

def plot_line2(df):
    plt.figure()
    if 'labels' in df.columns:
        plt.plot(df['labels'],df['value'],color=df['color'][0])
        plt.ylim((0,max(df['value'])*1.1))
        plt.xticks(rotation=45)
    else: 
        plt.plot(df['value'],color=df['color'][0])
        plt.ylim((0,max(df['value'])*1.1))
    fig = plt.gcf()
    plot_img = fig2img(fig)
    plot_img = np.array(plot_img*255., dtype='uint8')
    return plot_img

def img_processing(img, methods):
    cv2.imwrite('./test0.png', img)
    
    # ChartReader
    data = test(img, methods, 'KPGrouping')
    
    type_list = []
    
    for i in data:
        type_list.append(i[-1])

    graph_type = Counter(type_list)
    
    if not graph_type:
        return print("그래프 요소가 없습니다.")
    
    graph_type = max(graph_type, key=graph_type.get)
    print(graph_type)
    
    if graph_type ==0:
        is_dist, chart_data = bar_detection(img,data)
        df = pd.DataFrame(chart_data)
        plot_img1 = plot_bar1(img, df, is_dist)
        plot_img2 = plot_bar2(df)
    if graph_type ==1:
        is_dist, chart_data = line_detection(img,data[0])
        df = pd.DataFrame(chart_data)
        plot_img1 = plot_line1(df, is_dist)
        plot_img2 = plot_line2(df)
    elif graph_type ==2:
        is_dist, chart_data, start_angle, sort_index, start_point, pie_mid = pie_detection(img, data)
        df = pd.DataFrame(chart_data)
        plot_img1 = plot_pie1(img, df, start_point, pie_mid, is_dist)
        plot_img2 = plot_pie2(df,start_angle)
    
    csv_file = df[['labels','value']].to_csv(index=False)
    
    if plot_img1 is not None:
        _, plot_img1=cv2.imencode('.jpg', plot_img1, [int(cv2.IMWRITE_JPEG_QUALITY),90])
        cv2.imwrite('./test1.png', cv2.imdecode(plot_img1, 1))
        
        
    _, plot_img2=cv2.imencode('.jpg', plot_img2, [int(cv2.IMWRITE_JPEG_QUALITY),90])
    cv2.imwrite('./test2.png', cv2.imdecode(plot_img2, 1))
    
    return is_dist, plot_img1, plot_img2, csv_file

def download(url, params={}, data={}, headers={}, method='GET', retries=3):
    # if not canFetch(url, url):
    #     print('수집하면 안됨')
    
    resp = request(method, url, params=params, data=data, headers=headers)
    
    try:
        resp.raise_for_status()
    except HTTPError as e:
        if 499 < resp.status_code and retries > 0:
            print('재시도 중')
            sleep(5)
            return download(url, params, data, headers, method, retries-1)
        else:
            print(e.response.status_code)
            print(e.request.headers)
            print(e.response.headers)
        return None
        
    return resp

def get_img(url):

    # url로 부터 이미지 데이터 다운
    resp = download(url, headers={'user-agent':ua})

    if resp is None:
        print("에러")
        
        return

    # 이미지 코드 일시 content 데이터 사용 아닐시 함수 종료
    if re.search(r'image\/(\w+);?', resp.headers['content-type']):
        # ext = re.search(r'image\/(\w+);?', resp.headers['content-type']).group(1)
        # fname = re.search(r'/(\w+\.jpg|jpeg|png|bmp|gif)', url).group(1)
        
        # 이미지 디코드
        img = np.fromstring(resp.content, dtype='uint8')
        img = cv2.imdecode(img,1)
        return img
    
    else:
        return

def test(image, methods, model_type):
    # 使用 PyTorch 的 torch.no_grad() 上下文管理器来禁用梯度计算，以提高推理速度并减少内存使用。
    with torch.no_grad():
        # 使用预加载的 'KPDetection' 方法（存储在 methods 字典中）对图像进行关键点检测。methods['KPDetection'][2] 是测试函数，methods['KPDetection'][0] 是数据库对象，methods['KPDetection'][1] 是神经网络对象。
        results = methods[model_type][2](image, methods[model_type][0], methods[model_type][1])
        
        if model_type == 'KPGrouping':
            keys, centers, group_scores = results
            # 与 'KPDetection' 相同，但这里没有应用阈值过滤。

            keys = {k: [p for p in v.tolist()] for k,v in keys.items()} 
            centers = {k: [p for p in v.tolist()] for k,v in centers.items()}

            groups = get_groups(keys, centers, group_scores)
            
            return groups
        
async def handler(websocket):

    try:
        # 데이터 수신
        data = await websocket.recv()
        data = json.loads(data)
        
        if data:
            if 'imageUrl' in data.keys():
                url = data['imageUrl']
                img = get_img(url)
                
                if img is None:
                    print('이미지아님')
                    await websocket.send(json.dumps('이미지아님'))
                else:
                    is_dist, plot_img1, plot_img2, csv_file = img_processing(img, methods)
                    
                    
                    if is_dist and plot_img1 is not None:
                        plot_img1 =  base64.b64encode(plot_img1)
                        plot_img2 =  base64.b64encode(plot_img2)

                        send_data = {'distortionDetected':is_dist,
                                    'correctionMark':plot_img1.decode(),
                                    'correctionCopy':plot_img2.decode(),
                                    'csvFile': csv_file}
                        await websocket.send(json.dumps(send_data))
                    else:
                        plot_img2 =  base64.b64encode(plot_img2)

                        send_data = {'distortionDetected':is_dist,
                                    'correctionCopy':plot_img2.decode(),
                                    'csvFile': csv_file}
                        await websocket.send(json.dumps(send_data))
                    
            elif 'image' in data.keys():
                img = np.fromstring(base64.b64decode(re.sub('data:image/jpeg;base64,','',data['image'])), dtype='uint8')
                img = cv2.imdecode(img,1)
                
                if img is None:
                    print('이미지아님')
                    await websocket.send(json.dumps('이미지아님'))
                else:
                    is_dist, plot_img1, plot_img2, csv_file = img_processing(img, methods)
                    
                    
                    if is_dist and plot_img1 is not None:
                        plot_img1 =  base64.b64encode(plot_img1)
                        plot_img2 =  base64.b64encode(plot_img2)

                        send_data = {'distortionDetected':is_dist,
                                    'correctionMark':plot_img1.decode(),
                                    'correctionCopy':plot_img2.decode(),
                                    'csvFile': csv_file}
                        await websocket.send(json.dumps(send_data))
                    else:
                        plot_img2 =  base64.b64encode(plot_img2)

                        send_data = {'distortionDetected':is_dist,
                                    'correctionCopy':plot_img2.decode(),
                                    'csvFile': csv_file}
                        await websocket.send(json.dumps(send_data))
        else:
            print('데이터 비어있음')
            
    except websockets.ConnectionClosedOK:
        pass

    await websocket.close()


# 서버 시작
async def main(HOST, PORT):
    async with websockets.serve(handler, HOST, PORT, max_size=2**23):
        await asyncio.Future()  # run forever
        
if __name__ == "__main__":
    
    # ChartRedaer 옵션
    args = parse_args()
    args.save_path = "evaluation"
    args.model_type = "KPGrouping"
    args.data_dir = "./img/"
    args.cache_path = "./cache/"
    args.trained_model_iter = "best"
    
    methods = pre_load_nets(args.model_type, args.data_dir, args.cache_path, args.trained_model_iter)
    print(f"Predicting with {args.model_type} net")
    
    HOST = '127.0.0.1'
    PORT = 8080

    print('>> Server Start with ip :', HOST)

    asyncio.run(main(HOST,PORT))




