PYTHON = python
build_dir = build

laf:
	$(PYTHON) setup.py build_ext
	$(eval compiled_output := ${build_dir}/lib.*/*.so)
	@echo "Cython compilation output: $(compiled_output)"
	@echo "Success"

clean:
	rm -r laf.cpp build/
