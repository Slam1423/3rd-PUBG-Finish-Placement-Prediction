# PUBG-Finish-Placement-Prediction

![image](https://github.com/yansun1996/PUBG-Finish-Placement-Prediction/blob/master/src/FinalRun/PUBG%20Inlay.jpg)

This repository is our solution for Kaggle Competition: [PUBG Finish Placement Prediction](https://www.kaggle.com/c/pubg-finish-placement-prediction) (Kernel Only)

Our group chose this kaggle match(2018) as our final project for class "Big Data Analytics". 

The goal of this match is using some features in a game to predict each player's placement in that game.

# Feature engineering

We construct totally 247 features from the player level and the game level by excavating second order variables as well as statistics.

# Model selection

We chose three ML methods: xgboost, LightGBM and Multi-Layer Perceptron to train the prediction model. 

After this step, the mae(valuation) reaches about 0.022, ranking about 300/1500.

# Postprocessing

We found that the original variable killPlace reveals significant some ground truth information about the final placement. And we combine the topological sort and the above trained models to make a much more precise prediction.

After this step, the mae reaches 0.020, ranking 69/1500.

# Missing groups

There're many groups missing a game, but we have to give prediction to them as well. And we use GMM to fit the distribution of the mssing groups' placement and then insert into the final prediction by sampling.

After this step, the mae reaches 0.01528, ranking 3/1500.

# Conclusion

I think the most important step is Postprocessing. Because usually we make predictions straightly after finishing training a ML model to see how well it works. But if there's some useful and unconspicuous implication about out prediction goal, we have to be enough cute to catch them and combine them into our model, which could really make a big difference to our prediction.

# Our group

![image]()
