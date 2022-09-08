import re
import numpy as np
import json
import csv
import logging
import datetime
from dateutil.parser import parse,parserinfo

def_date = datetime.datetime(1700, 1, 1, 0, 0, 0)
aggr_tok = ['total', 'sum', 'max', 'min', 'avg', 'average', 'mean']  # words that indicate aggregate rows

# customized parserinfo for datetime parser
class CustomParserInfo(parserinfo):
    JUMP = [' ']

def det_dialect(file):
    """
    Find the csv dialect.
    """
    sniffer = csv.Sniffer()
    sample_bytes = 1024
    try:
        dialect = sniffer.sniff(open(file).read(sample_bytes))
        delim = dialect.delimiter
    except:
        print('Could not determine delimiter')
        delim = ';'
    print('Delimiter :', delim)
    return (delim)


def isNumeric(g):
    """
    Checks if the processed value is numeric.
    Identifies also the demical and thousand separator style, comma or period.
    """
    if re.findall("([1-9]/.[0-9]{3}){1,}", g) and not re.findall("([1-9]/.[0-9]{4}){1,}", g):  # eg. 1.000.000
        g = g.replace('.', '')
        g = g.replace(',', '.')
    elif re.findall("(,[0-9]{1,2}$)|(,[0-9]{4,}$)|^0*,", g):  # comma -> demical | eg. 1.000,05 ,  0,453
        g = g.replace('.', '')
        g = g.replace(',', '.')
    else:  # eg 1,000.04  1,000
        g = g.replace(',', '')
    try:
        g = float(g)
    except:
        return (g, 'text')
    return g, 'numeric'


def primaryKey(dict_data):
    """
    Identifies if an attributed can be considered as primary key.
    """
    for col in dict_data:
        actual_len = len(dict_data[col])
        unique_len = np.unique(np.array(dict_data[col])).shape[0]
        if actual_len == unique_len:
            return col
    return ''


def single_attr_freq(attr):
    """
    Returns the data type of the attribute and checks for data type consistency.
    """
    inc = False
    values, counts = np.unique(attr, return_counts=True)
    dictc = {}
    for ind, val in enumerate(values):
        dictc[val] = counts[ind]
    keys = dictc.keys()
    if len(keys) > 2:  # 2 or more data types for an attribute -> handle as text type and inconsistency
        dictc = {'text': sum(counts)}
        inc = True
    elif len(keys) == 2:
        if 'unknown' in keys:  # 'unknown' means NA value
            dictc.pop('unknown', None)
        else:
            dictc = {'text': sum(counts)}
            inc = True
    else:
        if 'unknown' in keys:  # empty column
            dictc['numeric'] = dictc['unknown']
            del dictc['unknown']
    attr_freq_dict = list(dictc.keys())[0]
    return attr_freq_dict, inc


def checkTimeDate(d):
    """
    Check if the value is date , time or timestamp.
    """
    if (d.time() == datetime.time(0, 0, 0, 0)):
        k = [d.date(), 'date']
    elif (d.date() == datetime.date(1700, 1, 1)):
        k = [d.time(), 'time']
    else:
        k = [d, 'timestamp']
    return k


def datatype_identify(table_data):
    """
    Processes the whole table(string values), identifies the data type of each column
    and makes the appropriate modifications e.g. if numeric -> float, if date-> datetime.date ...
    Returns the modified data, the data type for each attribute and where type inconsistency exists .
    """
    dict_data = {}
    dict_attr = {}
    inconsistent_cols = []
    for i, td in enumerate(table_data):
        cl = table_data[td].tolist()
        column_data = []
        column_attr = []
        for ic, cell in enumerate(cl):
            try:
                if np.isnan(cell):
                    k = [np.nan, 'unknown']   # 'unknown' means missing value (NA)
                else:
                    k = [cell, 'numeric']
            except:
                cell = cell.strip()
                extra = ''.join(re.findall("(^\D+|\D+$)", cell)).replace(' ', '')
                if not extra or extra == '$':  # e.g. 1445$, $115
                    k = isNumeric(cell.replace(extra, ''))
                    if type(k[0]) == str:
                        try:
                            d = parse(cell, default=def_date)
                            k = checkTimeDate(d)
                        except:
                            k = [cell, 'text']
                else:
                    try:
                        d = parse(cell, default=def_date,parserinfo=CustomParserInfo())
                        k = checkTimeDate(d)
                    except:
                        k = [cell, 'text']
            column_data.append(k[0])
            column_attr.append(k[1])
            if k[1] == 'text':
                if k[0].lower() in aggr_tok:
                    print('aggregation_token - ' + k[0] + ' - in line: ', str(ic))
                    column_data.pop(ic)
                    column_attr.pop(ic)
                    table_data.drop([ic], axis=0, inplace=True)
        type_col, inconsistency = single_attr_freq(column_attr)
        if inconsistency:   # if inconsistent -> text type and all values becomes strings
            print (td, 'inconsistent')
            inconsistent_cols.append(td)
            for id, d in enumerate(column_data):
                if column_attr[id] in ['numeric', 'date','time', 'time_stamp']:
                    column_data[id] = str(d)
        dict_data[td] = column_data
        dict_attr[td] = type_col

    return dict_data, dict_attr, inconsistent_cols


def parse_pytheas_annotation(annot_file, df_csv, csv_file_name):
    """
    Open and read Pytheas output (json file) and store the values in a dictionary.
    """
    f = open(annot_file)
    pytheas_out = json.load(f)
    tables = pytheas_out['tables']

    tables_dict = {}
    # [csv_file_name+'_table_'+str(ind)]
    for ind, t in enumerate(tables):
        table = {}
        attr_list = []
        try:
            table['df_metadata'] = list(df_csv.iloc[t['top_boundary']:t['header'][t['table_counter'] - 1], 0].dropna())
        except:
            print("No metadata detected! ")
            table['df_metadata'] = None
        try:
            table['df_foot'] = list(df_csv.iloc[t['footnotes'][0]:t['footnotes'][-1] + 1, 0])
        except:
            print("No footnotes detected!\n ")
            table['df_foot'] = None

        for h in t['columns']:
            # print (h)
            attr = ''
            for c in t['columns'][h]['column_header']:
                attr = attr + c['value'] + ''
            if not attr:
                attr = 'Column' + str(h)
            #attr = attr.replace(' ', '_')
            attr_list.append(f'{attr}')
        table['attt_names'] = attr_list
        table_data = df_csv.iloc[t['data_start']:t['data_end'] + 1, :].reset_index(drop=True).dropna(axis=1, how='all')
        table_data.columns = attr_list
        table['table_data'] = table_data
        table['conf'] = t['confidence']
        table['subheaders'] = t['subheaders']
        tables_dict[csv_file_name + '_table' + str(ind)] = table
    f.close()
    return tables_dict


def column_properties(series):
    CAT_FRAC_THRESHOLD = 0.5
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    cat_N_threshold = {"object": 1000, "int64": 10, "float64": 10}

    name = series.name
    colresult = {}
    colresult["dtype"] = str(series.dtype)
    nulls = series.isnull().sum()
    colresult["nulls"] = int(nulls) if not np.isnan(nulls) else 0
    notnulls = series.dropna()

    colresult["notnulls"] = len(notnulls.index)
    colresult["numeric"] = (
            series.dtype in [np.float64, np.int64] and colresult["notnulls"] > 0
    )
    unique = notnulls.unique().size
    colresult["unique"] = unique
    colresult["is_categorical"] = False
    if (
            colresult["dtype"] in {"object", "int64", "float64"}
            and colresult["notnulls"] > 0
    ):
        # In Pandas integers with nulls are cast as floats, so we have
        # to include floats as possible categoricals to detect
        # categorical integers.
        colresult["is_categorical"] = (
                                              unique / colresult["notnulls"] <= CAT_FRAC_THRESHOLD
                                      ) and (unique <= cat_N_threshold[colresult["dtype"]])
        logger.debug(
            "Column {:15}: {:6} unique, {:6} notnulls, {:6} total"
            " --> {}categorical".format(
                name,
                unique,
                colresult["notnulls"],
                colresult["notnulls"] + colresult["nulls"],
                "NOT " * (not colresult["is_categorical"]),
            )
        )


    colresult["is_ID"] = False

    return {name: colresult, "_columns": [name]}





