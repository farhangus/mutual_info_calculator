# Mutual Score Calculator

This repository contains a Python script for calculating the mutual information between entropy values and various features from a z-score matrix. The mutual information values help identify the dependency between the entropy and the features.

## Requirements

- Python 3.x
- pandas
- scikit-learn

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/mutual-score-calculator.git
    cd mutual-score-calculator
    ```

2. Install the required Python libraries:
    ```sh
    pip install pandas scikit-learn
    ```

3. Ensure you have the input CSV files:
    - `240501_01_entropy-values.csv`
    - `240603_01_mtx_motif_zscore.csv`

## Usage

1. Place the input CSV files in the same directory as the script.
2. Run the script:
    ```sh
    python mutual_score_calculator.py
    ```

## Output

- The script outputs a CSV file named `final_result.csv` which contains the mutual information values for each feature, sorted in descending order.

## Optimization Details

- The script reads and transposes the z-score matrix only once to minimize file I/O operations.
- Unnecessary reshaping and conversions of the data are removed to enhance performance.
- The use of pandas and scikit-learn ensures efficient handling and computation of the data.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For any questions or suggestions, please contact [yourname](mailto:youremail@example.com).

