# TensorNets

## Requirements
To install requirements, set up a virtual environment and use the following command from project root:

	pip install -r extras/includes/requirements.txt

Due to dependency contradictions, you may have to install gym and supersuit individually outside of the above command, specifying the versions given at the bottom of the requirements file!!!

## Running tests
To run the tests simply go into the APP directory and run the following command

	python -m unittest test_main.TestModels

Doing this will allow you to make sure that every package is installed, and that your system will run training, model transfer and model predictions without issue.
