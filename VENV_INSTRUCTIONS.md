# Python Virtual Environment Setup Guide

## What is a Virtual Environment?

A virtual environment is an isolated Python environment that allows you to install packages without affecting your system Python installation. This is especially useful for projects that have specific package requirements.

## Quick Setup (Automated)

Run the setup script:

```bash
bash setup_venv.sh
```

This will:
1. Create a virtual environment named `venv`
2. Activate it
3. Install all required packages (pandas, openpyxl, reportlab)

## Manual Setup

### Step 1: Create Virtual Environment

```bash
python3 -m venv venv
```

This creates a new directory called `venv` containing the virtual environment.

### Step 2: Activate Virtual Environment

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

When activated, you'll see `(venv)` at the beginning of your command prompt.

### Step 3: Install Required Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install pandas openpyxl reportlab
```

### Step 4: Verify Installation

```bash
pip list
```

You should see pandas, openpyxl, and reportlab in the list.

## Using the Virtual Environment

### Activate (when you start working):
```bash
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate      # Windows
```

### Deactivate (when you're done):
```bash
deactivate
```

### Run Your Scripts

Once activated, run your Python scripts normally:
```bash
python3 main.py
python3 generate_pdf.py
```

## Project Structure

After setup, your project will look like:
```
Final Project/
├── venv/                 # Virtual environment (don't edit this)
├── main.py
├── generate_pdf.py
├── result.txt
├── requirements.txt
├── setup_venv.sh
└── ...
```

## Troubleshooting

### If `python3 -m venv` doesn't work:
- Make sure Python 3 is installed: `python3 --version`
- On some systems, use `python` instead of `python3`

### If activation doesn't work:
- Make sure you're in the project directory
- Check that `venv/bin/activate` exists (Linux/Mac) or `venv\Scripts\activate` exists (Windows)

### If packages don't install:
- Make sure the virtual environment is activated (you should see `(venv)` in your prompt)
- Try upgrading pip: `pip install --upgrade pip`

## Important Notes

- **Always activate the virtual environment** before running your scripts
- The `venv` folder should be added to `.gitignore` if you're using git (it's large and system-specific)
- You can recreate the environment anytime by deleting `venv` and running setup again




