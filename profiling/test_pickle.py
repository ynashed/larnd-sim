import pickle
from .records import Records

# with open('memprof_20230105-140206.pkl', 'rb') as f:
with open('/home/pgranger/larnd-sim/clean_repo/larnd-sim/results/memprof_clear_cache_nograd_notes.pkl', 'rb') as f:
	obj = pickle.load(f)

code_infos = obj['code_infos']
rec = Records(obj['line_records'], code_infos)

funcs = [elt['func'] for elt in code_infos.values()]

base_columns = ('active_bytes.all.peak', 'reserved_bytes.all.peak')

for func in funcs:
	print(rec.display(func, base_columns))