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
    """
    Parse the header of an MNRAS table.

    Args:
        row (BeautifulSoup object): A row of the HTML table header.
        **kwargs: Additional arguments to pass to `format_mnras_header`.

    Returns:
        list: A list of formatted column headers.
    """    
    text = list(filter(None,row.text.split('\n            .\xa0')))    

    return format_mnras_header(text, **kwargs)

def format_mnras_header ( columns, lower=True ):
    """
    Format MNRAS column headers by replacing special characters and optionally converting to lowercase.

    Args:
        columns (list): A list of column header strings.
        lower (bool): Whether to convert the column headers to lowercase (default is True).

    Returns:
        list: A list of formatted column headers.
    """    
    if lower:
        columns_reformatted = [ col.lower() for col in columns]
    else:
        columns_reformatted = columns
        
    for orig, new in replace_dict.items():
        columns_reformatted = [ col.replace(orig,new) for col in columns_reformatted ]
    return columns_reformatted

def parse_mnras_row ( row ):
    """
    Parse a single row of an MNRAS table body.

    Args:
        row (BeautifulSoup object): A row of the HTML table body.

    Returns:
        list: A list of strings representing the row's cell values.
    """    
    items = []
    pattern =  r"\s×\s10<sup>(−?\d+)</sup>"
    for element in row.find_all('td'):    
        text = str(element)
        result = re.sub(pattern, r"e\1", text).strip('<td>').strip('</td>')
        result = re.sub(r'\s','', result)
        items.append(result)
    return items


def parse_mnras_body ( data ):
    """
    Parse the body of an MNRAS table.

    Args:
        data (BeautifulSoup object): The HTML table element containing the body.

    Returns:
        list: A list of rows, where each row is a list of strings representing cell values.
    """    
    body = data.find('tbody')
    rows = body.find_all('tr')
    body_formatted = []
    for row in rows:
        body_formatted.append(parse_mnras_row(row))
    return body_formatted

def read_mnras_table ( filename, require_header=True ):    
    """
    Read an MNRAS table from an HTML file and convert it into a pandas DataFrame.

    Args:
        filename (str): The path to the HTML file containing the MNRAS table. (Currently assumes that the HTML file is saved locally.)
        require_header (bool): Whether to require a valid header (default is True).

    Returns:
        pandas.DataFrame: A DataFrame containing the table data, with headers and data types inferred.
    """    
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
        if require_header:
            raise ValueError (f"Read {len(colnames)} column names, but found {len(body[0])} columns!\nColumns: {colnames}")
        else:
            colnames = np.arange(len(body[0]))
    
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