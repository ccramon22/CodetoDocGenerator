# ParameterPilots

A project for fine-tuning language models to generate documentation for Python code.

## Project Overview

This project aims to train a language model to automatically generate documentation for Python functions. It includes:
- Data collection from popular Python repositories
- Preprocessing of Python code and docstrings
- Model training and fine-tuning
- Evaluation of generated documentation

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git
- CUDA-compatible GPU (recommended for training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/33AndrewM33/ParameterPilots.git
cd ParameterPilots
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/MacOS
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script to train the model and generate documentation:
```bash
python main.py
```

The script will:
- Clone Python repositories
- Extract function-docstring pairs
- Train the documentation generation model
- Evaluate the model on test cases

## Project Structure

- `main.py`: Main script for running the entire pipeline
- `data_collection.py`: Functions for cloning and processing repositories
- `preprocessing.py`: Code for extracting and preparing function-docstring pairs
- `model.py`: Model training and fine-tuning implementation
- `evaluation.py`: Functions for evaluating model performance
- `analytics.py`: Analytics and performance tracking
- `data/`: Directory for storing processed datasets
- `github_repos/`: Directory for cloned repositories
- `training_results/`: Directory for storing model checkpoints and results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

Copyright (c) 2024 ParameterPilots Team

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

**Important Notice**: This software and its associated models, datasets, and documentation are protected by copyright law. Unauthorized copying, distribution, modification, public display, or public performance of copyrighted works is an infringement of the copyright holders' rights. As the owner of the copyright in this work, we reserve all rights not expressly granted to you. 