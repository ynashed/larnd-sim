"""Class and helper calls for processing and displaying line records"""
import inspect
import pandas as pd
from math import isnan


def best_repr(nbytes):
    units = [
        (1024 ** 5, 'P'),
        (1024 ** 4, 'T'),
        (1024 ** 3, 'G'),
        (1024 ** 2, 'M'),
        (1024 ** 1, 'K'),
        (1024 ** 0, 'B'),
    ]

    abs_nbytes = abs(nbytes)
    sign = -1 if nbytes < 0 else 1

    for factor, unit in units:
        if nbytes >= factor:
            best_factor, best_unit = factor, unit
            break

    value = sign*nbytes/factor

    return f"{value:.2f}{unit}" if not isnan(value) else ''

def make_lalign_formatter(df, cols=None):
    """
    https://stackoverflow.com/a/67202912
    Construct formatter dict to left-align columns.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The DataFrame to format
    cols : None or iterable of strings, optional
        The columns of df to left-align. The default, cols=None, will
        left-align all the columns of dtype object

    Returns
    -------
    dict
        Formatter dictionary

    """
    if cols is None:
       cols = df.columns[df.dtypes == 'object'] 

    return {col: f'{{:<{df[col].str.len().max()}s}}'.format for col in cols}


class Records:
    def __init__(self, records, calls):
        code = []
        for code_hash in calls.keys():
            func = calls[code_hash]['func']
            lines, start_line = inspect.getsourcelines(func)
            lines = list(zip([code_hash]*len(lines), [func.__qualname__]*len(lines), lines, range(start_line, start_line + len(lines))))
            code = code + lines
        self._code = pd.DataFrame(code, columns = ['code_hash', 'func_name', 'code', 'line'])
        self._records = pd.DataFrame(records)
        self._merged = pd.merge(self._records, self._code, on=['code_hash', 'line'], how='right')
        self._merged.prev_record_idx = self._merged.prev_record_idx.astype('Int64')

        self._accumulate_records()
        
        

    def display(self, func, columns):
        if len(self._merged) == 0:
            return 'No data collected\n'

        is_byte_col = self._merged.columns.get_level_values(0).str.contains('byte')
        byte_cols = self._merged.columns[is_byte_col]

        string = {}
        selected = self._merged[self._merged.func_name == func.__qualname__].copy()

        #Only displaying the max for each
        selected = selected.groupby(['code_hash', 'line'], as_index=False).max()
        
        maxlen = max(len(c) for c in selected.code)
        left_align = '{{:{maxlen}s}}'.format(maxlen=maxlen)
        selected[byte_cols] = selected[byte_cols].applymap(best_repr)
        selected = selected[[*columns, 'line', 'code']]
        selected['code'] = selected['code'].apply(lambda l: l.rstrip('\n\r'))



        string[func.__qualname__] = selected.to_string(index=False, justify='left', formatters=make_lalign_formatter(selected, cols=['code']))

        return '\n\n'.join([f"{q:*^80}\n\n{c}\n\n{'*'*80}\n\n" for q, c in string.items()])


    def _accumulate_records(self):
        acc_mask = self._merged.columns.str.match(r'.*(allocated|freed)$')
        peak_mask = self._merged.columns.str.match(r'.*(peak)$')
        acc_raw, peak_raw = self._merged.loc[:, acc_mask].values, self._merged.loc[:, peak_mask].values
        acc_corr, peak_corr = acc_raw.copy(), peak_raw.copy()

        for row, record in self._merged.iterrows():
            if pd.isna(record['prev_record_idx']) or record['prev_record_idx'] == -1 or record['prev_record_idx'] == row-1:
                continue

            acc_corr[row] = acc_raw[record['prev_record_idx']+1:row+1].sum(0)
            peak_corr[row] = peak_raw[record['prev_record_idx']+1:row+1].max(0)

        corr = self._merged.copy()
        corr.loc[:, acc_mask] = acc_corr
        corr.loc[:, peak_mask] = peak_corr
        return corr