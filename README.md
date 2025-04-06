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
In a specific match, the home and away teams' OPR and DPR are pitted against each other with a home/away correction in order to come up with predicted goals for each side. The specific formula for predicting actual goals from the Power Rankings is below:
<br/><br/>
home_predG = (home_team.OPR - (xG_avg - away_team.DPR))*home_correction
away_predG = (away_team.OPR - (xG_avg - home_team.DPR))/away_correction
<br/><br/>
Historically in the Premier League the home team has scored 16% more goals than average, and the away team has scored 13% fewer goals than average. This equates to about 30% more goals expected for a team playing at home vs playing away, and shows how home field is a big deal in soccer (biggest home field advantage across all major sports, interestingly).
<br/><br/>
The predicted goal number for each team is a decimal number, and a Poisson distribution is used to come up with the probability of that team scoring 0, 1, 2... goals (this analsis goes up to 6 goals). Doing that for both teams and assuming the goals scored by each team are independent of each other, a matrix of every possible score outcome is generated with the probability for each scoreline, which looks like the figure below from 538:
<figure>   
  <img src="/images/ScoreMatrix.png" width="400" height="400">
    <figcaption><center>Scoreline probability matrix, from 538.</center></figcaption>
</figure>
