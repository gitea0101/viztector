# viztector
[ChartReader](https://github.com/zhiqic/ChartReader), [korean_ocr_using_pororo](https://github.com/black7375/korean_ocr_using_pororo)
기반 그래프 왜곡탐지 모델

### 콘다 가상환경 설정

~~~
conda create -n Viztector python=3.7.13
conda activate Viztector
~~~

### 패키지 및 Pororo 설치

~~~
pip install -r .\requirements.txt
git clone https://github.com/kakaobrain/pororo.git
cp ./pororo_setup.py ./pororo/setup.py
cd pororo
pip install -e .
~~~

cache/nnet/KPGrouping 안에 KPGrouping_best.pkl 파일 설치  
https://drive.google.com/file/d/11Z6cNl-5dcpbuDq9N_ZrB7rPUoJvAH_u/view?usp=drive_link  
  
### 모델 테스트  

img/images/val 안에 테스트 이미지 복사후  
detection_test.ipynb 실행  
  
### 왜곡 탐지  

correction_test.ipynb 실행  

### 익스텐션 사용
서버 실행  
~~~
python server.py  
~~~
크롬 익스텐션 설치  
~~~
확장 - 확장 관리 - 압축 풀린 파일 로드 - Viztector_CE 폴더 선택
Image and Video Capture Extension
Start 버튼 클릭후 이미지 클릭
~~~

Start 버튼 이미지 클릭시 서버로 이미지 전송  
Capture 버튼 Start 버튼 클리후 활성 웹페이지내 Video를 캡쳐하여 서버로 이미지 전송  
Reset 버튼 이벤트 리스너 및 저장된 CSV 데이터 삭제  
download 버튼 서버에 업로드된 데이터 CSV파일로 다운로드  
