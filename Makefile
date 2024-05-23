VENV_DIR = venv

PIP = $(VENV_DIR)/bin/pip

REQUIREMENTS_FILE = requirements.txt

all: install

$(VENV_DIR): # Create a virtual environment
	python3 -m venv $(VENV_DIR)

install: $(VENV_DIR) # Create a virtual environment and install the requirements.txt
ifeq (,$(wildcard $(REQUIREMENTS_FILE)))
	@echo "File $(REQUIREMENTS_FILE) not found."
else
	$(PIP) install -r $(REQUIREMENTS_FILE)
endif
	@echo "Run the following command to source the virtual environment 'source venv/bin/active'"

save:
	$(PIP) freeze > $(REQUIREMENTS_FILE)

clean:
	rm -rf $(VENV_DIR)

.PHONY: all install clean save