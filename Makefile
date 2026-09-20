# KDAgent - one-command pipeline
#
# Enable optional steps:
#   make sample-data         prepare the reproducible sample-case file
#   make build-rag           build the SWaT RAG knowledge base (needs real data)
#   make run                 run the batch experiment across all configured models
#   make smoke               run the mocked end-to-end smoke test (no real API)
#   make unit                run the offline unit tests
#   make check               syntax-compile every source file
#   make install             install pinned dependencies (reproducible)
#
# Every `python ...` step sets PYTHONPATH so `src.*` imports resolve from the
# project root regardless of the current working directory.

PYTHON ?= python
export PYTHONPATH := .

.PHONY: install sample-data build-rag run smoke unit check help

help:
	@echo "Targets: install, sample-data, build-rag, run, smoke, unit, check"

install:
	$(PYTHON) -m pip install -r requirements-lock.txt

sample-data:
	$(PYTHON) scripts/make_sample_data.py

build-rag:
	$(PYTHON) src/build_rag_index.py \
		--data_dir data \
		--kb_dir data/rag_kb \
		--rag_config configs/rag.yaml \
		--rebuild

# Full experiment: pass extra args with ARGS="--models model_a --max_cases 10"
run:
	$(PYTHON) src/run_all_models.py $(ARGS)

smoke:
	$(PYTHON) scripts/smoke_test.py

unit:
	$(PYTHON) src/test_iterative_agent.py && \
	$(PYTHON) src/test_dual_branch_fusion_agent.py && \
	$(PYTHON) src/test_model_client_budget.py && \
	$(PYTHON) src/test_final_experiment_runner.py && \
	$(PYTHON) src/test_rag_kb_builder.py

check:
	$(PYTHON) -m compileall -q src scripts experiments analysis