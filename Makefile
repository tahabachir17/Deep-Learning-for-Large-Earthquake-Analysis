.PHONY: test train-case-i train-case-ii train-case-iii evaluate clean

test:
	python -m pytest -q

train-case-i:
	python scripts/train_model.py --x-path data/GNSS_M3S_181/xdata.npy --y-path data/GNSS_M3S_181/ydata.npy --nst 3 --nt 181 --output-dir reports/case_i

train-case-ii:
	python scripts/train_model.py --x-path data/GNSS_M7S_181/xdata.npy --y-path data/GNSS_M7S_181/ydata.npy --nst 7 --nt 181 --output-dir reports/case_ii

train-case-iii:
	python scripts/train_model.py --x-path data/GNSS_M7S_501/xdata.npy --y-path data/GNSS_M7S_501/ydata.npy --nst 7 --nt 501 --output-dir reports/case_iii

evaluate:
	python scripts/evaluate_model.py --data-root data/real_events --model-case-i checkpoints/GNSS_M3S_181/model_Standard.h5 --model-case-ii checkpoints/GNSS_M7S_181/model_Standard.h5 --output-csv reports/real_data_results.csv

clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', 'reports/case_i', 'reports/case_ii', 'reports/case_iii']]"
