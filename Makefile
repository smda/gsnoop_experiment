.PHONY: all
all: run

# Run custom commands
.PHONY: run
run:
	@echo "[`date +\"%T\"`] Installing dependencies..."
	pip3 install --break-system-packages --upgrade -r requirements.txt >/dev/null 2>/dev/null
	@mkdir -p build/params/
	@mkdir -p results/

	@echo "[`date +\"%T\"`] Generating synthetic oracles..."
	@python3 scripts/generate.py

	@echo "[`date +\"%T\"`] Time for some coffee..."
	@cp scripts/run_experiment.py ./run_experiment.py

# Clean target to remove specific files or directories
.PHONY: clean
clean:
	@echo "Cleaning up..."
	@rm -rf build/ 
	@rm -rf run_experiment.py

