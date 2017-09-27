name: inverse
class: center, middle, inverse
layout: true
title: Python-basic

---
class: titlepage, no-number

# Python Science toolkit
## .gray.author[Youngjae Yu]

### .x-small[https://github.com/yj-yu/python-scipy]
### .x-small[https://yj-yu.github.io/python-scipy]

.bottom.img-66[ ![](images/snu-logo.png) ]

---
layout: false

## About

- Review on recent deep learning projects
- Python numpy, scipy : tutorials and enhanced references
- All you need for ML : Matplotlib, pandas, sklearn etc..

---

template: inverse

# Let's imagine about our future work!
advance of deep learning applications

---

template: inverse
# 앞으로 실제 실험해 볼 주제를 생각해야 합니다.

---

## How to catch up Deep Learning

Stanford cs231n lectures
- http://cs231n.stanford.edu/syllabus.html

DNN, CNN, RNN 등의 강의 내용은 앞으로
cs231n을 부교재로 보충하시면 좋습니다!

그리고 이 내용들을 참고하세요
- https://github.com/sjchoi86/Tensorflow-101
- https://github.com/hunkim/DeepLearningZeroToAll
- https://github.com/golbin/TensorFlow-ML-Exercises

---

## Applications - Go

.center.img[ ![](images/alphago_99_800-800x533.jpg) ]

---

## Applications - color restoration

.center.img[ ![](images/lettherebecolor1-800x1014.png) ]

---

## Applications - cucumber classification

.center.img[ ![](images/cucumber-farmer-9.png) ]

---

## Applications - private secretary

.center.img[ ![](images/kakao-mini-1-491x295.png) ]

---

## Applications - translate animal's language

.center.img[ ![](images/Flickr.ume-y.CC-BY-2.0-800x533.jpg) ]

---

## Applications - Save animals

.center.img[ ![](images/find_the_sea_cow_solution-800x532.jpg) ]

---

## Applications - Save animals

.center.img[ ![](images/dugong-800x600.jpg) ]

---

## Applications - Natural disaster prediction

.center.img[ ![](images/Flickr.UniversityofSalfordPressOffice.CC-BY-2.0.jpg) ]

---

## Applications - Medical

.center.img[ ![](images/mcgill-800x426.jpg) ]

---

## Applications - Medical

.center.img[ ![](images/Flickr.Joyce-Kaes.CC-BY-2.0-800x533.jpg) ]

---

## Applications - Music

.center.img[ ![](images/magenta-800x390.png) ]

---

## Applications - others

.center.img[ ![](images/entrupy-800x446.png) ]

---

## Google Lens

https://youtu.be/igTtOA1jcik

## Video fraud

https://youtu.be/MVBe6_o4cMI
https://lyrebird.ai/demo/

---


template: inverse
## Python - Scipy


---
## Install configuration

```python
git clone https://github.com/yj-yu/python-scipy.git
cd python-basic
ls
```

code(https://github.com/yj-yu/python-scipy)

```bash
./code
├── pandas_matplotlib.ipynb
├── pycon-2017
└── scipy-2017-sklearn

```

- pandas_matplotlib : https://github.com/rabernat/python_teaching
- pycon-2017 : https://github.com/tylerjereddy/pycon-2017/
- scipy-2017-sklearn : https://github.com/amueller/scipy-2017-sklearn

---
If you use anaconda, use conda to install scipy
```python
conda install scipy
pip install scikit-learn
pip install scikit-image
pip install pandas
pip install matplotlib
```

```python
cd code
jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000
```

Open pandas_matplotlib.ipynb

---


template: inverse
## Python - Plot your data


---

## There are many libraries based on matplotlib

Googling!

seaborn
- https://seaborn.pydata.org/

```python
import seaborn; seaborn.set()
from matplotlib import pyplot as plt
%matplotlib inline
```


---

## And you can build your dataset on database environment

pandas
- http://pandas.pydata.org/

```python
import pandas as pd
```

---

## Interactive jupyter notebook

bokeh 
- interactive visualization library
- https://bokeh.pydata.org/en/latest/


```python
from bokeh.plotting import figure, show, output_notebook
output_notebook()
```

---

template: inverse
## Python - Scipy


---

```python
cd code
cd scipy-2017-sklearn/notebook
jupyter notebook
```


---


name: last-page
class: center, middle, no-number
## Thank You!


<div style="position:absolute; left:0; bottom:20px; padding: 25px;">
  <p class="left" style="margin:0; font-size: 13pt;">
  <b>Special Thanks to</b>: 변훈 연구원님, 송재준 교수님</p>
</div>

.footnote[Slideshow created using [remark](http://github.com/gnab/remark).]




<!-- vim: set ft=markdown: -->
