# PremierLeague_PowerRatings_and_Predictions
Live offensive and defensive power ratings for each of the 20 teams are used to predict goals, win/draw/loss probabilities, and upcoming schedule difficulty.
<figure>   
  <img src="/images/EPL24_OPR_GW31.png" width="800" height="500">
    <figcaption><center>Offensive Power Ratings, gameweek 31.</center></figcaption>
</figure>

<figure>   
  <img src="/images/EPL24_DPR_GW31.png" width="800" height="500">
    <figcaption><center>Defensive Power Ratings, gameweek 31 (lower = better defense).</center></figcaption>
</figure>

## Table of Contents
1. Overview
2. Dependencies
3. Methodology & Results
4. Further Work

## 1. Overview
This project maintains Offensive Power Ratings (OPR) and Defensive Power Ratings (DPR) for each of the 20 Premier League teams which correspond to the average number of goals you would predict the team to score and concede against an average team on a neutral field, similar to the 538 predictions [here](https://fivethirtyeight.com/methodology/how-our-club-soccer-predictions-work/). For each matchup, the two teams' power ratings are used to predict home and away goals, win/draw/loss probabilities, and upcoming schedule difficulty.
<br/><br/>
After each match, the predictions are compared to the match results and the power ratings adjusted up or down according to adjusted goals (adjG), a weighted combination of actual goals and expected goals (xG). This allows the model to take into account the fixture difficulty. For example, if a team were predicted to score 2.1 goals in a game and they actually generated 2.8 adjG, their OPR would be adjusted upwards for their next fixture.

## 2. Dependencies
* Python 3

## 3. Methodology & Results
