import requests
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

replace_dict = {
    '⊙':'odot',
    '#':'n',
    '\xa0':'_',
    ' ':'_'
}

def parse_mnras_header (row, **kwargs):
    text = list(filter(None,row.text.split('\n            .\xa0')))    

    return format_mnras_header(text, **kwargs)

def format_mnras_header ( columns, lower=True ):
    if lower:
        columns_reformatted = [ col.lower() for col in columns]
    else:
        columns_reformatted = columns
        
    for orig, new in replace_dict.items():
        columns_reformatted = [ col.replace(orig,new) for col in columns_reformatted ]
    return columns_reformatted

def parse_mnras_row ( row ):
    items = []
    pattern =  r"\s×\s10<sup>(−?\d+)</sup>"
    for element in row.find_all('td'):    
        text = str(element)
        result = re.sub(pattern, r"e\1", text).strip('<td>').strip('</td>')
        result = re.sub(r'\s','', result)
        items.append(result)
    return items


def parse_mnras_body ( data ):
    body = data.find('tbody')
    rows = body.find_all('tr')
    body_formatted = []
    for row in rows:
        body_formatted.append(parse_mnras_row(row))
    return body_formatted

def read_mnras_table ( filename ):    
    response = open(filename, 'r')
    soup = BeautifulSoup(response.read(), 'html.parser')
    data = soup.find('table')
    
    colnames = parse_mnras_header(data.find('thead').find_all('tr')[0])
    colunits = parse_mnras_header(data.find('thead').find_all('tr')[1], lower=False)
    body = parse_mnras_body(data)

    dl = len(body[0]) - len(colnames)
    if dl == 1:
        colnames = ['rowid'] + colnames
    elif dl == 0:
        pass
    else:
        raise ValueError    
    
    df = pd.DataFrame(body, columns=colnames)
    for col in df.columns:
        is_all_integers = df[col].apply(lambda x: x.isdigit()).all()
        if is_all_integers:
            df[col] = df[col].astype(int)
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass
        
            
    return df