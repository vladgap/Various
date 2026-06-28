import numpy as np
import pandas as pd

tools_version = '3.0'
print(f' Version of tools is {tools_version}\n',
      'Remove class NN2to1 -- moved to SHLN')

def CopyPasteToPandas(a):
    if a.startswith('\n'):
        a = a[1:]
    if a.endswith('\n'):
        a = a[:-1]
    rows = a.split('\n')
    data = []
    for row in rows:
        parts = row.split('\t')
        parsed_row = []
        for item in parts:
            item = item.strip()
            try:
                # נסה להמיר ל-int או float
                if '.' in item:
                    parsed_row.append(float(item))
                else:
                    parsed_row.append(int(item))
            except ValueError:
                parsed_row.append(item)
        data.append(parsed_row)
    return data

def CopyPasteToArray(a):
    if a.startswith('\n'):
        b=a[1:]
    else:
        b=a
    if b.endswith('\n'):
        b=b[:-1]
    c=b.replace('\t',',')
    d=c.split('\n')
    f=[]
    for e in d:
        if e.replace(',','').replace('.','').isdigit(): # only digits no letters
            f.append(list(eval(e)))
        else:
            f.append(e.split(','))
    return f


def round_to_significant_digits(num, keep_int=False):
    if keep_int and isinstance(num, int):
        return num
    if not isinstance(num, (int, float)):
        return num  # Return non-numeric types as is
    abs_num = abs(num)
    if abs_num >= 100:
        return round(num)    # Existing rules for numbers less than 100
    elif abs_num >= 0.1:
        return float(f'{num:.3g}')        # 3 significant digits
    elif 0.01 <= abs_num < 0.1:
        return float(f'{num:.2g}')        # 2 significant digits
    else: # abs_num < 0.01
        return float(f'{num:.1g}')        # 1 significant digit

def apply_rounding_to_structure(data_structure, rounding_func, **kwargs):
    if isinstance(data_structure, pd.DataFrame):
        # For DataFrames, map is used, and kwargs can be passed directly
        return data_structure.map(lambda x: rounding_func(x, **kwargs))
    elif isinstance(data_structure, list):
        rounded_list_of_lists = []
        for inner_list in data_structure:
            rounded_inner_list = []
            for item in inner_list:
                rounded_inner_list.append(rounding_func(item, **kwargs))
            rounded_list_of_lists.append(rounded_inner_list)
        return rounded_list_of_lists
    else:
        raise TypeError("Input must be a pandas DataFrame or a list of lists")
