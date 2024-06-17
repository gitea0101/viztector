# viztector


cache/nnet/KPGrouping 안에 KPGrouping_best.pkl 파일 설치 

https://drive.google.com/file/d/11Z6cNl-5dcpbuDq9N_ZrB7rPUoJvAH_u/view?usp=drive_link

img/images/val 안에 테스트 이미지 

~~~python
python val_extraction.py \
    --save_path evaluation \
    --model_type KPGrouping \
    --data_dir "./img/" \
    --cache_path "./cache/" \
    --trained_model_iter "best"
~~~

또는 detection_test.ipynb 실행

correction_test.ipynb 로 전체 실행
