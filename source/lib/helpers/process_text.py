import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
import re
from datetime import datetime

def clean_text(text, normalize=True, lower=True, remove_whitespace=True, remove_html=True, remove_urls=True):
    if isinstance(text, str):
        if not text:
            return pd.NA
        
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii') if normalize else text
        text = re.sub(r'<.*?>', '', text) if remove_html else text
        text = re.sub(r'(https?://\S+|www\.\S+)', '', text) if remove_urls else text
        text = re.sub(r'[\n\t\s]+', ' ', text) if remove_whitespace else text
        text = text.lower() if lower else text
        text = text.strip()
        
        if not text:
            return pd.NA
        
        return text
    
    elif isinstance(text, pd.Series):
        text = text.fillna('')
        text = text.str.normalize('NFKD').str.encode('ascii', 'ignore').str.decode('ascii') if normalize else text
        text = text.str.replace(r'<.*?>', '', regex=True) if remove_html else text
        text = text.str.replace(r'(https?://\S+|www\.\S+)', '', regex=True) if remove_urls else text
        text = text.str.replace(r'[\n\t\s]+', ' ', regex=True) if remove_whitespace else text
        text = text.str.lower() if lower else text
        text = text.str.strip()
        text = text.replace('', pd.NA)
        return text
    
    else:
        raise TypeError("Input must be a string or a pandas Series.")


def clean_date(date, output_format="%Y-%m-%d", aggregation=None):
    MONTH_MAP = {
        "jan": "01", "january": "01",
        "feb": "02", "february": "02",
        "mar": "03", "march": "03",
        "apr": "04", "april": "04",
        "may": "05",
        "jun": "06", "june": "06",
        "jul": "07", "july": "07",
        "aug": "08", "august": "08",
        "sep": "09", "september": "09",
        "oct": "10", "october": "10",
        "nov": "11", "november": "11",
        "dec": "12", "december": "12"
    }
    PATTERNS = {
        "dd/mm/yyyy": r'(\d{1,2})/(\d{1,3})/(\d{2,4})',
        "dd/mon/yyyy": r'(\d{1,2})/([a-z]{3})/(\d{4})',
        "dd-mon-yyyy": r'(\d{1,2})-([a-z]{3})-(\d{4})',
        'month dd, yyyy': r'\s*([a-z]{3,9})\s*(\d{1,2}),\s*(\d{4})\s*',
        'mm/yyyy': r'(\d{2})/(\d{4})',
        'mm-yyyy': r'(\d{2})-(\d{4})',
        'month, yyyy': r'([a-z]{3,9}),\s*(\d{4})',
        'yyyymm': r'(\d{4})(\d{2})'
    }
    if aggregation and aggregation not in ["day", "month", "year"]:
        raise ValueError("`Aggregation` must be either 'day', 'month', 'year', or `None`.")
    
    if isinstance(date, str):
        date = clean_text(str(date))
        if not date:
            return pd.NA
        format = None
        
        for pattern, regex in PATTERNS.items():
            if re.match(regex, date):
                format = pattern
        
        if format in ["dd/mm/yyyy", "dd/mm/yy"]:
            day, month, year = re.match(PATTERNS[format], date).groups()
            month = MONTH_MAP[month]
            result = pd.to_datetime(f"{year}-{month}-{day.zfill(2)}", errors='coerce')
        
        elif format in ["dd/mon/yyyy", "dd-mon-yyyy"]:
            day, month, year = re.match(PATTERNS[format], date).groups()
            month = MONTH_MAP[month]
            result = pd.to_datetime(f"{year}-{month}-{day.zfill(2)}", errors='coerce')
        
        elif format in ["month dd, yyyy"]:
            month, day, year = re.match(PATTERNS[format], date).groups()
            month = MONTH_MAP[month]
            result = pd.to_datetime(f"{year}-{month}-{day.zfill(2)}", errors='coerce')
        
        elif format in ["mm/yyyy", "mm-yyyy", "month, yyyy"]:
            month, year = re.match(PATTERNS[format], date).groups()
            result = pd.to_datetime(f"{year}-{month}-01", format="%Y-%m-%d", errors='coerce')       
        
        elif format in ["yyyymm"]:
            year, month = re.match(PATTERNS[format], date).groups()
            result = pd.to_datetime(f"{year}-{month}-01", format="%Y-%m-%d", errors='coerce')
        
        else:
            return pd.NA
        
        if aggregation == "year":
            result = result.replace(month=1, day=1)
        elif aggregation == "month":
            result = result.replace(day=1)
        
        return result
    
    elif isinstance(date, pd.Series):
        date = clean_text(date.astype("string"))
        
        test_date = None
        for val in date:
            if pd.notna(val) and val:
                test_date = val
                break
        print("Test date: ", test_date)
        if not test_date:
            return pd.Series(pd.NA, dtype='object')
        
        format = None
    
        for pattern, regex in PATTERNS.items():
            if re.match(regex, test_date):
                format = pattern
        print("Format: ", format)
        
        if format in ["dd/mm/yyyy", "dd/mm/yy"]:
            extracted = date.str.extract(PATTERNS[format])
            extracted.columns = ["day", "month", "year"]
            result = pd.to_datetime(extracted["year"] + "-" + extracted["month"] + "-" + extracted["day"].str.zfill(2), format=output_format, errors="coerce")

        elif format in ["dd/mon/yyyy", "dd-mon-yyyy"]:
            extracted = date.str.extract(PATTERNS[format])
            extracted.columns = ["day", "month", "year"]
            extracted["month"] = extracted["month"].map(MONTH_MAP)
            result = pd.to_datetime(extracted["year"] + "-" + extracted["month"] + "-" + extracted["day"].str.zfill(2), format=output_format, errors="coerce")
        
        elif format in ["month dd, yyyy"]:
            extracted = date.str.extract(PATTERNS[format])
            extracted.columns = ["month", "day", "year"]
            extracted["month"] = extracted["month"].map(MONTH_MAP)
            result = pd.to_datetime(extracted["year"] + "-" + extracted["month"] + "-" + extracted["day"].str.zfill(2), format=output_format, errors="coerce")
        
        elif format in ["mm/yyyy", "mm-yyyy", "month, yyyy"]:
            extracted = date.str.extract(PATTERNS[format])
            extracted.columns = ["month", "year"]
            result = pd.to_datetime(extracted["year"] + "-" + extracted["month"] + "-01", format=output_format, errors="coerce")
        
        elif format in ["yyyymm"]:
            extracted = date.str.extract(PATTERNS[format])
            extracted.columns = ["year", "month"]
            result = pd.to_datetime(extracted["year"] + "-" + extracted["month"] + "-01", format=output_format, errors="coerce")
        
        else:
            return pd.Series(pd.NA, dtype='object')
        
        if aggregation == "year":
            result = result.dt.to_period('Y')
        elif aggregation == "month":
            result = result.dt.to_period('M')
        
        return result
        
    else:
        raise TypeError("Input must be a string or a pandas Series.")


