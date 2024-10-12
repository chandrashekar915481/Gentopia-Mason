from gentopia.tools import BaseTool
from pydantic import Field
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import base64
import io
import yfinance as yf
import os
import asyncio

class GoldPricePredictor(BaseTool):
    name: str = "gold_price_predictor"
    description: str = "Predict gold prices based on recent data using Random Forest regression"
    
    gold_data: pd.DataFrame = Field(default=None)
    gold_model: RandomForestRegressor = Field(default=None)

    def __init__(self):
        super().__init__()

    def load_data(self, file_path):
        print(f"Loading gold data from CSV: {file_path}")
        self.gold_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        self.preprocess_data()

    def preprocess_data(self):
        print("Preprocessing data")
        self.gold_data['days'] = (self.gold_data.index - self.gold_data.index.min()).days
        self.gold_data['month'] = self.gold_data.index.month
        self.gold_data['year'] = self.gold_data.index.year

    def prepare_features(self):
        X = self.gold_data[['days', 'month', 'year', 'Open', 'High', 'Low', 'Volume']]
        y = self.gold_data['Close']
        return X, y

    def train_model(self):
        print("Training gold model with hyperparameter tuning...")
        X, y = self.prepare_features()
        
        param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
        
        grid_search.fit(X, y)
        
        self.gold_model = grid_search.best_estimator_
        print(f"Best parameters for gold model: {grid_search.best_params_}")

    def make_prediction(self, days_ahead):
        print(f"Making prediction for {days_ahead} days ahead")
        last_day = self.gold_data['days'].max()
        last_date = self.gold_data.index.max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead)
        future_data = pd.DataFrame({
            'days': range(last_day + 1, last_day + days_ahead + 1),
            'month': future_dates.month,
            'year': future_dates.year,
            'Open': [self.gold_data['Open'].iloc[-1]] * days_ahead,
            'High': [self.gold_data['High'].iloc[-1]] * days_ahead,
            'Low': [self.gold_data['Low'].iloc[-1]] * days_ahead,
            'Volume': [self.gold_data['Volume'].iloc[-1]] * days_ahead
        }, index=future_dates)
        predictions = self.gold_model.predict(future_data)
        return pd.DataFrame({'date': future_dates, 'predicted_price': predictions})

    def plot_forecast(self, forecast):
        print("Creating plot")
        plt.figure(figsize=(12, 6))
        plt.plot(self.gold_data.index, self.gold_data['Close'], label='Historical Gold')
        plt.plot(forecast['date'], forecast['predicted_price'], label='Gold Forecast', linestyle='--')
        plt.title('Gold Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        return image_base64

    def format_prediction_results(self, results):
        formatted_output = f"""
Gold Price Prediction Results:
==============================
Current Gold Price: ${results['current_gold_price']:.2f}

Forecast for next 3 days:
-------------------------
"""
        for prediction in results['gold_forecast']:
            formatted_output += f"Date: {prediction['date']}, Predicted Price: ${prediction['predicted_price']:.2f}\n"
        
        formatted_output += f"""
Model Performance:
------------------
Mean Absolute Error: {results['gold_mae']:.2f}
Root Mean Square Error: {results['gold_rmse']:.2f}

A plot of the forecast has been generated but cannot be displayed in this console output.
"""
        return formatted_output

    def _run(self, file_path: str, days_ahead: int = 3):
        try:
            self.load_data(file_path)
            self.train_model()
            
            # Make predictions
            gold_forecast = self.make_prediction(days_ahead)
            plot_base64 = self.plot_forecast(gold_forecast)

            # Calculate error metrics
            X, y = self.prepare_features()
            gold_y_true = y[-days_ahead:]
            gold_y_pred = self.gold_model.predict(X[-days_ahead:])

            gold_mae = mean_absolute_error(gold_y_true, gold_y_pred)
            gold_rmse = np.sqrt(mean_squared_error(gold_y_true, gold_y_pred))

            results = {
                'current_gold_price': self.gold_data['Close'].iloc[-1],
                'gold_forecast': gold_forecast.to_dict(orient='records'),
                'plot': plot_base64,
                'gold_mae': gold_mae,
                'gold_rmse': gold_rmse
            }
            
            return self.format_prediction_results(results)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return f"An error occurred: {str(e)}"

    async def _arun(self, file_path: str, days_ahead: int = 3):
        return await asyncio.to_thread(self._run, file_path, days_ahead)