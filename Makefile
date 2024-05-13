.PHONY: all
all: run

# Run custom commands
.PHONY: run
run:
	@pip3 install --upgrade --force-reinstall -r requirements.txt  >/dev/null 2>/dev/null
	@mkdir -p build/params/
	@mkdir -p results/

	@python3 scripts/generate.py

	@cp scripts/submit.sh ./submit.sh
	@chmod +x submit.sh

	@cp scripts/run_experiment.py ./build/run_experiment.py

# Clean target to remove specific files or directories
.PHONY: clean
clean:
	@echo "Cleaning up..."
	@rm -rf *.out
	@rm -rf build/ 
	@rm -rf results/
	@rm -rf submit.sh 
	@rm -rf run_experiment.py 
