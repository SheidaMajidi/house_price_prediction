# house_price_prediction

The house price prediction project aimed to construct a robust multi-task learning model capable of performing both regression (predicting house prices) and classification (predicting house categories). This comprehensive analysis involved several crucial steps, from initial data exploration to hyperparameter tuning. Below is a detailed review of each step undertaken in the project.

Link to Datasets: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

### Data Exploration and Preprocessing

An in-depth initial analysis was conducted to grasp the dataset's structure and deal with missing values, utilizing statistical summaries and visualizations such as histograms and heatmaps.

#### In the preprocessing phase:

- Numerical columns were imputed using KNN imputer, ensuring the robust handling of missing values.
- Categorical columns were populated with the most frequent category, preserving the dataset's integrity.
- Feature Engineering: Crafted features like HouseAge and HouseCategory, derived from YearBuilt, YearRemodAdd, and BldgType.
- Pipeline Setup: Imputation was done using SimpleImputer, with a 'median' strategy for numerical and 'most_frequent' for categorical columns. OrdinalEncoder was used to encode categorical features, and MinMaxScaler for feature scaling.

### Multi-task Model Building

The constructed model shares a common feature extraction base, splitting into two heads tailored for individual tasks: one for price prediction (using MSE loss) and another for house category classification (using Cross-Entropy loss).

### Activation Functions and Optimizers

A series of experiments were conducted with activation functions—ReLU, LeakyReLU, Sigmoid, Tanh—paired with optimizers—Adam, SGD, RMSprop. These combinations were evaluated based on the model's performance metrics.

### Loss Functions

Employed a dual loss function approach:

- Regression: Mean Squared Error Loss
- Classification: Cross-Entropy Loss
- Model Evaluation

The final model metrics were noteworthy, achieving:

- Mean Absolute Error (MAE): 60811.4087
- Accuracy: 46.58%
- Advanced PyTorch Lightning Features

I incorporated TensorBoard for tracking and ModelCheckpoint callbacks to automatically save the top-performing models.

### Hyperparameter Tuning

Using Optuna, I discovered the optimal learning rate (0.003490200025511474) and initial neuron count (74), which significantly minimized the validation loss.

### Model Persistence

The well-tuned model is saved as:

- optimized_model_full.pth - Full PyTorch model
- optimized_model.pkl - Pickle format for compatibility
