import uvicorn
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
from tensorflow.keras.models import load_model


app = FastAPI()

model = load_model('../sentiment_model')


class Input(BaseModel):
    text: str


@app.get('/')
def index():
    return {'message': 'Welcome to the Sentiment Analysis API'}


@app.post('/predict')
def predict_sentiment(data: Input):
    """ FastAPI 
    Args:
        data: json file 
    Returns:
        sentiment: Sentiment of the text - (Negative or Positive)
        prediction: probability of text being positive
    """
    data = data.dict()
    review = data['text']
    prediction = model.predict([review])
    prediction_prob = prediction.tolist()[0][0]

    pred_map = {0: 'Negative', 1: 'Positive'}

    return {
        'sentiment': pred_map[round(prediction_prob)],
        'probability': prediction_prob
    }


# Run the app
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
