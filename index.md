### Introduction

Our project involves comparing the competitive regions of North America and Western Europe
in the popular online competitive game League of Legends. Fans from both regions have commonly
bantered back and forth as to which region is the superior one, and thus, using the online
Kaggle League of Legends dataset (found [here](https://www.kaggle.com/chuckephron/leagueoflegends)) this
project hopes to add fuel to the raging debate.

The Kaggle League of Legends dataset includes the data of every single competitive game of League of
Legends played from Summer 2014 to Summer 2017. While the dataset contains a vast amount of detailed
information about each game, the only data we will be concerning ourselves with is winrate. After all,
it doesn't matter how well your individual players perform in a game if the game is still lost. A team could
be massively ahead, then embarrassingly lose the overall match or series. Thus, the only valid metric for
comparison would be winrate.

Unfortunately, the dataset does not contain any information about things not directly related to the game.
Thus we cannot see team payrolls, staff size, or how long a team has been around. Those metrics would be
interesting to analyze, but we do not have the data to perform such analysis.

We start out by importing everything necessary for the project. 

'import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.linear_model import LinearRegression'
