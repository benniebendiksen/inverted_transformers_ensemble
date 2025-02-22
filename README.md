# Inverted_transformers_ensemble
A Technical Analysis employing test of the Unicorn Binance Trailing Stop Loss Engine within Binance's Futures Exchange

## Installation
To facilitate the installation of dependencies, in an OS agnostic way, non-native libraries relied upon by this project will be provided in the form of requirements.txt and setup.py files, respectively. Once completed, this project will be uploaded to PyPI (Python Package Index).

### Virtual Environment Creation
cd into the root directory and create/load a Python virtual environment:
```
$ cd trend-activated-trailing-stop-loss-bot
$ python3 -m venv tslb_env
$ source tslb_env/bin/activate
```
Having created this venv without an environment,yml file, you will need to install the non-native libraries relied upon by this project, found within the requirements.txt or setup.py files:
```
$ cd ..
$ python3 -m pip install -r requirements.txt 
or
$ python3 setup.py install
```

### Conda Environment Creation
Alternatively, open up a command line interface and from the root directory of the project run:
```
conda env create -f environment.yml
```
NOTE: This approach may lead to a ResolvePackageNotFound error depending on your machine specifications.

This will attempt to create the conda environment for you with the correct dependencies and versions installed. Then run:
```
conda activate trend_activated_bot_env
```
This will activate the conda environment for you. 

## Execute
Ensure your Binance Futures-enabled API KEYS are set as OS environment variables.
You are able to now run the project!
```
$ export API_KEY="aaa"
$ export API_SECRET="bbb"
$ ./main.py
```
