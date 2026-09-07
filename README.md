# CSC6621-Final-Project

This is all done in WSL or Powershell

Create the venv

```
python -m venv .venv
or
python3 -m venv .venv
```

Activate the venv with the following

wsl
```
source .venv/bin/activate
```

PowerShell
```
.\.venv\bin\Activate.ps1
or
.\.venv\Scripts\Activate.ps1
```

Run the following to install all the necessary packages
```
pip install -U pip wheel # Updates pip and wheel
pip install -r requirements.txt # Installs the libraries
```

## Running the Jupyter Notebook
Run the following to use the Jupyter Notebook
```
jupyter notebook
```

## Checking for diff
Once you install the requirements you can run the following to make git diffs of the notebooks more readable
```
nbdime config-git --enable --global
```
Note this is mostly for git on the terminal
