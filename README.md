
# DataBase Management Systems | Project 

## Abstract
Comma Separated Value (CSV) is one of the most
important Open Data formats extensively used in a plethora
of domains because of his simplicity and flexibility in storing
data. However, data published in this format often does not
comply to strict specifications, rising a variety of ambiguities
and making the automated data extraction a challenging task.
Although the relational data (table) extraction from CSV files
has not been studied extensively, there is a novel framework
named Pytheas which can effectively manage this challenge.
Nevertheless, Pytheas focuses on table discovery which is just
an intermediate key step in the CSV processing pipeline. In this
work, I deal with the remaining steps, mainly with data type
detection one, in order to convert the CSV file in a proper format
for data analysis purposes and storing in the relational database
management system PostgreSQL. Moreover, I build a dashboard
which gives to users a clear data insight and enables them to store
and


## Installation instructions

The following instructions have been tested on a newly created Windows 10 with Python3.7.12 on a conda environment. Create your conda environment by running conda create --name pytheas-venv python==3.7.12 , 
and activate it by running conda activate pytheas-venv. 
Clone the repo to your machine using git git clone https://github.com/JKaraman93/Pytheas.git .

First, you have to setup and run Pytheas, and then the Extra Component.


## Pytheas #### (Original repo: https://github.com/cchristodoulaki/Pytheas)

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
For more details : https://github.com/JKaraman93/Pytheas/edit/master/src/README.md


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



