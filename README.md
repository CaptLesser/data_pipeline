# data_pipeline

A Python-based cryptocurrency quant research toolkit and pipeline.

## Features
- Fetches top 24h "losers" (biggest price drops) on Kraken (see `kraken_losers.py`)
- (More modules coming soon...)

## Setup

1. Clone this repo:
    ```sh
    git clone https://github.com/CaptLesser/data_pipeline
    cd data_pipeline
    ```

2. Install requirements:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

- **Top Losers Script:**
    ```sh
    python kraken_losers.py
    ```
Outputs a CSV and prints top 24h price drops to the terminal.

## Contributing

Pull requests and issues welcome!
