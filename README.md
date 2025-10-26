# AWS ELB Request Anomaly Detection using an Autoencoder

This project uses a Keras/TensorFlow Autoencoder to identify anomalous traffic patterns in an AWS Elastic Load Balancer (ELB) request count dataset.

The goal is to build an unsupervised learning model capable of learning what "normal" traffic looks like. Data points that the model struggles to reconstruct (i.e., those with a high reconstruction error) are flagged as anomalies.

The notebook also demonstrates the use of the **Optuna** library to optimize the autoencoder's hyperparameters and find the most efficient model architecture.

## Methodology

The project workflow follows these steps:

1.  **Data Loading and Cleaning:**
    * The dataset (`elb_request_count_8c0756.csv`) is loaded using Pandas.
    * Extreme outliers are removed using Z-score (`scipy.stats.zscore`) to stabilize the dataset.

2.  **Preprocessing:**
    * The `value` column (request count) is isolated and normalized using `MinMaxScaler` from Scikit-learn, placing all values in the [0, 1] range. This is essential for training neural networks.

3.  **Hyperparameter Optimization (Optuna):**
    * A function `create_autoencoder` is defined for use by Optuna.
    * Optuna tests 10 combinations (`n_trials=10`) of hyperparameters to find the best configuration, aiming to minimize the `val_loss` (Mean Squared Error).
    * The optimized parameters are:
        * `encoding_dim` (dimension of the latent layer)
        * `learning_rate`
        * `batch_size`

4.  **Final Model Training:**
    * A new Keras `Sequential` autoencoder is built using the best hyperparameters found by Optuna.
    * The final model is compiled with the Adam optimizer and the `mse` loss function.
    * The model is trained on the scaled data (`X_scaled`) for 50 epochs.

5.  **Anomaly Detection:**
    * The trained model is used to predict (reconstruct) the input data.
    * The reconstruction error (MSE) is calculated for each data point.
    * An **anomaly threshold** is set at the **80th percentile** of the errors. This means the 20% of data with the highest reconstruction error are classified as anomalies.
    * Data is marked as anomalous (`anomaly = True`) if its reconstruction error is above this threshold.

6.  **Visualization:**
    * A `matplotlib` chart is generated showing the original time series, with the data points identified as anomalies highlighted in red.

## Technologies Used

* **Python 3**
* **TensorFlow / Keras:** To build and train the autoencoder model.
* **Optuna:** For hyperparameter optimization.
* **Pandas:** For data manipulation and cleaning.
* **NumPy:** For numerical calculations.
* **Scikit-learn:** For data preprocessing (MinMaxScaler).
* **Matplotlib:** For visualizing the results.
* **Scipy:** For Z-score calculation.
* **Jupyter Notebook:** As the development environment.

## Dataset

The project uses the `elb_request_count_8c0756.csv` file, which contains time-series data of request counts from an AWS ELB. The main columns are `timestamp` and `value`.

## Results

The script identifies and flags the data points that deviate most from the "normal" pattern learned by the autoencoder. At the end of the execution, the model classified **~19.3%** of the data as anomalous (based on the 80th percentile threshold).

The final output is a visualization of the time series with unusual traffic patterns clearly marked.

![Anomaly Plot](httpsAdditional-Files/anomaly_plot.png)
*(Recommendation: Run the notebook, save the generated image to a folder like `Additional-Files/`, commit it, and update this link.)*

## How to Run

1.  Clone this repository:
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd [REPOSITORY_NAME]
    ```

2.  Create a virtual environment and install the dependencies. It's recommended to create a `requirements.txt` file:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  Start Jupyter Notebook to run `train.ipynb`:
    ```bash
    jupyter notebook train.ipynb
    ```