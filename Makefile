.PHONY: test train train-case-i train-case-ii train-case-iii evaluate clean

test:
	python -m pytest -q

train: train-case-i train-case-ii train-case-iii

train-case-i:
	python scripts/train_model.py --config configs/case_i.yaml

train-case-ii:
	python scripts/train_model.py --config configs/case_ii.yaml

train-case-iii:
	python scripts/train_model.py --config configs/case_iii.yaml

evaluate:
	python scripts/evaluate_model.py --data-root data/real_events --model-case-i reports/case_i/model/model.keras --model-case-ii reports/case_ii/model/model.keras --output-csv reports/real_data_results.csv

clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', 'reports/case_i', 'reports/case_ii', 'reports/case_iii']]"
