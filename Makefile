.PHONY: test-parser

test-parser:
	pytest tests/test_parser.py

test-converters:
	pytest tests/test_converters.py