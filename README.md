# TSA-Time-series-exploration
This takes tsa data that I previously used in my previous tsa project and uses some time series techniques to try to predict how many people go through tsa in a week. I worked on this because there is a market in Kalshi for predicting TSA checkins and I thought it would be fun to build something simple to begin trying to model this. 

Right now this does not quite have the level of precision to bet with, it'll get you close though. It includes a rolling average, SARIMA model, prophet model, xgboost, and a hybrid model of xgboost and prophet. I also have a scrape script to help update the data. 

Future steps could explore differnt hyperparameters, I didn't use grid searchCV() for tuning hyperparameters for my xgboost and that seems like a natural next step. I think in the near future this could also be tried with other methods. 
