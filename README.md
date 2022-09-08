
# Installation instructions

The following instructions have been tested on a newly created Windows 10 with Python3.7.12 on a conda environment. Create your conda environment by running conda create --name pytheas-venv python==3.7.12 , 
and activate it by running conda activate pytheas-venv. 
Clone the repo to your machine using git git clone https://github.com/JKaraman93/Pytheas.git .

First, you have to setup and run Pytheas, and then the Extra Component.


## Pytheas

### Setup

Package the Pytheas project into a [wheel](https://realpython.com/python-wheels/), and install it using pip:
```
cd src
python setup.py sdist bdist_wheel
pip install  --upgrade --force-reinstall dist/pytheas-0.0.1-py3-none-any.whl
```

### Load trained weights and inference
There are pretrained Pytheas rules using a set of 2000 Open Data CSV files from Canadian CKAN portals.

E.g.:
```
cd pytheas
python ppytheas.py infer -w trained_rules.json -f ../../data/examples/demo6.csv -o inferred_annotation.json
```
If you want, you can also train Pytheas using your own files and annotations.
For more details : https://github.com/JKaraman93/Pytheas/edit/master/src/README_pytheas.md


# Extra Component

### Install requirements

```
cd Extra_Component
pip install -r requirements.txt
```
### Inference

The Component uses the Pytheas output file i.e., inferred_annotation.json. You only need to specify the CSV file as follows: 

```
python run.py -c ../pytheas/data/examples/demo6.csv
```

### Dashboard

To display the Dashboard, follow the link in terminal e.g Dash is running on http://127.0.0.1:8050/ .



