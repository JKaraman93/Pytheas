-------------------------
# Pytheas
-------------------------


## Installation instructions

The following instructions have been tested on a newly created Windows 10 with Python3.7.12 on a conda environment. Create your conda environment by running conda create --name pytheas-venv python==3.7.12 , 
and activate it by running conda activate pytheas-venv. 
Clone the repo to your machine using git git clone https://github.com/JKaraman93/Pytheas.git .


README_pytheas.md

### Setup

Package the Pytheas project into a [wheel](https://realpython.com/python-wheels/), and install it using pip:
```
cd src
python setup.py sdist bdist_wheel
pip install  --upgrade --force-reinstall dist/pytheas-0.0.1-py3-none-any.whl
```

#### Load trained weights and inference
There are pretrained Pytheas rules using a set of 2000 Open Data CSV files from Canadian CKAN portals.

E.g.:
```
cd pytheas
python ppytheas.py infer -w trained_rules.json -f ../../data/examples/demo6.csv -o inferred_annotation.json
```
* If you want, you can also train Pytheas using your own files and annotations.
For more details : https://github.com/JKaraman93/Pytheas/edit/master/src/README_pytheas.md
