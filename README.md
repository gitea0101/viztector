# viztector


cache/nnet/KPGrouping 안에

https://drive.google.com/file/d/11Z6cNl-5dcpbuDq9N_ZrB7rPUoJvAH_u/view?usp=drive_link

img/images/val 안에 이미지

~~~python
python val_extraction.py \
    --save_path evaluation \
    --model_type KPGrouping \
    --data_dir "./img/" \
    --cache_path "./cache/" \
    --trained_model_iter "best"
~~~
