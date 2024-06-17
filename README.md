# viztector
콘다 가상환경 설정
~~~
conda create -n Viztector python=3.7.13
conda activate Viztector
~~~

패키지 및 Pororo 설치
~~~
pip install -r .\requirements.txt
git clone https://github.com/kakaobrain/pororo.git
cp ./pororo_setup.py ./pororo/setup.py
cd pororo
pip install -e .
~~~

cache/nnet/KPGrouping 안에 KPGrouping_best.pkl 파일 설치  
https://drive.google.com/file/d/11Z6cNl-5dcpbuDq9N_ZrB7rPUoJvAH_u/view?usp=drive_link  
  
모델 테스트  
img/images/val 안에 테스트 이미지 복사후  
detection_test.ipynb 실행  
  
왜곡 탐지  
correction_test.ipynb 실행  
  
크롬 익스텐션 설치  
  
서버 실행  
python server.py  

