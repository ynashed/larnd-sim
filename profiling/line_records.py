"""Class and helper functions for processing and displaying line records"""
import inspect
import pandas as pd


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

    return f"{value:.2f}{unit}"


def _accumulate_line_records(raw_line_records):
    raw = pd.DataFrame(raw_line_records)
    acc_mask = raw.columns.str.match(r'.*(allocated|freed)$')
    peak_mask = raw.columns.str.match(r'.*(peak)$')
    acc_raw, peak_raw = raw.loc[:, acc_mask].values, raw.loc[:, peak_mask].values
    acc_refined, peak_refined = acc_raw.copy(), peak_raw.copy()

    for row, record in enumerate(raw_line_records):
        if record['prev_record_idx'] == -1:
            # No previous data to accumulate from
            continue
        if record['prev_record_idx'] == row-1:
            # Previous record was the previous line, so no need to accumulate anything
            continue

        # Another profiled function has been called since the last record, so we need to
        # accumulate the allocated/freed/peaks of the intervening records into this one.
        acc_refined[row] = acc_raw[record['prev_record_idx']+1:row+1].sum(0)
        peak_refined[row] = peak_raw[record['prev_record_idx']+1:row+1].max(0)

    refined = raw.copy()
    refined.loc[:, acc_mask] = acc_refined
    refined.loc[:, peak_mask] = peak_refined
    return refined


def _line_records(raw_line_records, code_infos):
    qual_names = {
        code_hash: info['func'].__qualname__ for code_hash, info in code_infos.items()}
    records = (_accumulate_line_records(raw_line_records)
               .assign(qual_name=lambda df: df.code_hash.map(qual_names))
               .set_index(['qual_name', 'line'])
               .drop(['code_hash', 'num_alloc_retries', 'num_ooms', 'prev_record_idx'], axis=1))
    records.columns = pd.MultiIndex.from_tuples(
        [c.split('.') for c in records.columns])

    return records


class LineRecords:
    def __init__(self, raw_line_records, code_infos):
        self._raw_line_records = raw_line_records
        self._code_infos = code_infos

    def display(self, func, columns):
        line_records = self._filter_raw_line_records(func, columns)
        return RecordsDisplay(line_records, self._code_infos)

    def _filter_raw_line_records(self, func, columns):

        if len(self._raw_line_records) == 0:
            return pd.DataFrame(index=pd.MultiIndex.from_product([[], []]), columns=columns)

        line_records = _line_records(self._raw_line_records, self._code_infos)
        line_records = _extract_line_records(line_records, func, columns)

        if len(line_records) > 0:
            line_records = line_records.groupby(level=[0, 1]).max()

        return line_records


def _extract_line_records(line_records, func, columns):
    if func is not None:
        # Support both passing the function directly and passing a qual name/list of qual names
        line_records = line_records.loc[[func.__qualname__] if callable(func) else func]

    if columns is not None:
        columns = [tuple(c.split('.')) for c in columns]
        if not all(len(c) == 3 for c in columns):
            raise ValueError('Each column name should have three dot-separated parts')
        if not all(c in line_records.columns for c in columns):
            options = ", ".join(".".join(c)
                                for c in line_records.columns.tolist())
            raise ValueError(
                'The column names should be fields of torch.cuda.memory_stat(). Options are: ' + options)
        line_records = line_records.loc[:, columns]

    return line_records


class RecordsDisplay:
    def __init__(self, line_records, code_infos):
        self._line_records = line_records
        self._code_infos = code_infos
        self._merged_line_records = self._merge_line_records_with_code()

    def _merge_line_records_with_code(self):
        merged_records = {}
        for _, info in self._code_infos.items():
            qual_name = info['func'].__qualname__
            if qual_name in self._line_records.index.get_level_values(0):
                lines, start_line = inspect.getsourcelines(info['func'])
                lines = pd.DataFrame.from_dict({
                    'line': range(start_line, start_line + len(lines)),
                    'code': lines})
                lines.columns = pd.MultiIndex.from_product([lines.columns, [''], ['']])

                merged_records[qual_name] = pd.merge(
                    self._line_records.loc[qual_name], lines,
                    right_on='line', left_index=True, how='right')
        return merged_records

    def __repr__(self):
        if len(self._line_records) == 0:
            return 'No data collected\n'

        is_byte_col = self._line_records.columns.get_level_values(0).str.contains('byte')
        byte_cols = self._line_records.columns[is_byte_col]

        string = {}
        for qual_name, merged in self._merge_line_records_with_code().items():
            maxlen = max(len(c) for c in merged.code)
            left_align = '{{:{maxlen}s}}'.format(maxlen=maxlen)
            merged[byte_cols] = merged[byte_cols].applymap(best_repr)

            # This is a mess, but I can't find any other way to left-align text strings.
            code_header = (left_align.format('code'), '', '')
            merged[code_header] = merged['code'].apply(lambda l: left_align.format(l.rstrip('\n\r')))
            merged = merged.drop('code', axis=1, level=0)

            string[qual_name] = merged.to_string(index=False)

        return '\n\n'.join(['## {q}\n\n{c}\n'.format(q=q, c=c) for q, c in string.items()])
