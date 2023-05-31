import pandas as pd
from .models.model import TitanicModel
from .data.preprocessing import DataPreprocessor

class TitanicCLI:
    """
    Handles the command-line interface operations for the Titanic Machine Learning Model. 
    This includes operations like training the model and making predictions.

    Attributes
    ----------
    model_path : str
        Path where the trained model is (or will be) stored
    preprocessor_path : str
        Path where the data preprocessor object is (or will be) stored
    """

    def __init__(self,  model_path: str, preprocessor_path: str):
        """
        Constructs all the necessary attributes for the TitanicCLI object.

        Parameters
        ----------
        model_path : str
            Path where the trained model is (or will be) stored
        preprocessor_path : str
            Path where the data preprocessor object is (or will be) stored
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

    def train(self, train_data: str):
        """
        Trains the Titanic ML Model using the given data and saves the trained model and preprocessor.

        Parameters
        ----------
        train_data : str
            Path to the CSV file containing the training data
        """
        # Load training data
        df = pd.read_csv(train_data)

        # Preprocess the data
        preprocessor = DataPreprocessor(df)
        preprocessor.prepare_data()
        preprocessor.feature_engineering()
        preprocessor.handle_missing_values()
        preprocessor.transform_data()
        # Train the model
        model = TitanicModel(5)
        model.train(preprocessor.train_df, preprocessor.train_y)

        # Save the model and preprocessor
        model.save(self.model_path)
        preprocessor.save(self.preprocessor_path)

    def predict(self, predict_data: str, out_path: str):
        """
        Makes predictions using the saved model and preprocessor on the given data and writes the predictions to the given path.

        Parameters
        ----------
        predict_data : str
            Path to the CSV file containing the data to make predictions on
        out_path : str
            Path where the predictions will be written
        """
        # Load the model and preprocessor
        model = TitanicModel.load(self.model_path)
        preprocessor = DataPreprocessor.load(self.preprocessor_path)

        # Load and preprocess the prediction data
        df = pd.read_csv(predict_data)
        df = preprocessor.prepare_data(df)
        df = preprocessor.feature_engineering(df)
        df = preprocessor.handle_missing_values(df)
        df = preprocessor.transform_data(df)

        # Make predictions
        predictions = model.predict(df)

        # Print predictions
        DataPreprocessor.save_predictions(out_path, df, predictions)