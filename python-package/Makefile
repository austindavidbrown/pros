develop:
	python3 setup.py check build_ext --inplace
	python3 test.py

pip_install:
	pip3 install ./
	python3 test.py

python_setup:
	virtualenv -p python3 ./env