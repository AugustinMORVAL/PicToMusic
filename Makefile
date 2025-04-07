.PHONY: test-parser, test-converters, streamlit

streamlit:
	@streamlit run app.py

test-parser:
	@pytest tests/test_parser.py

test-converters:
	@pytest tests/test_converters.py