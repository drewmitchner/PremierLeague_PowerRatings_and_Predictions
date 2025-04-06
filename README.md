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
### 3a. Match Predictions
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
<br/><br/>

Using this matrix, win, draw, and loss probabilities as well as clean sheet odds are calculated for each team. Then for every gameweek, a list of match summaries are generated:
<figure>   
  <img src="/images/MatchSummaries.png" width="300" height="500">
    <figcaption><center>Match summaries.</center></figcaption>
</figure>
<br/><br/>

From the summaries, a list is generated for the the top predicted scoring teams as well as the teams most likely to keep a clean sheet.
<figure>   
  <img src="/images/GW_PredictedGoals.png" width="200" height="335">
    <figcaption><center>Single gameweek predicted goals list.</center></figcaption>
</figure>
<br/><br/>
<figure>   
  <img src="/images/GW_PredictedCleanSheet.png" width="200" height="335">
    <figcaption><center>Single gameweek predicted clean sheets.</center></figcaption>
</figure>

### 3b. Power Rating Updates
Most importantly to this model, there is a correction step after observing the results of the games. If a team did better than expected (either scored more goals than predicted, or conceded fewer), their OPR and DPR are adjusted up or down accordingly, and vice versa for performing worse than expected.

Because goals in soccer are so rare and there's a lot of variance, the team's "performance" has additional precision beyond just the actual goals scored and conceded. Specifically a new statistic called adjusted goals (adjG) is calculated. This is the average of actual goals with a couple tweaks (discount goals scored against 10 men, discount late goals scored by a team already winning, etc.) and expected goals. Expected goals (xG) is based on historical data of many years worth of shots with records for where on the field they were taken, right foot/left foot, header/volley, defenders in the way, etc. and finds how many times a player scored from that kind of shot. So if a shot from a certain position was scored 10% of the time, then that shot would be awarded an expected goal (xG) of 0.1.
<br/><br/>

The two plots below show each team's OPR and DPR over the course of the season, up to current gameweek 31 in earl April.
<figure>   
  <img src="/images/EPL24_OPR_GW31.png" width="800" height="500">
    <figcaption><center>Offensive Power Ratings, gameweek 31.</center></figcaption>
</figure>

<figure>   
  <img src="/images/EPL24_DPR_GW31.png" width="800" height="500">
    <figcaption><center>Defensive Power Ratings, gameweek 31 (lower = better defense).</center></figcaption>
</figure>
<br/><br/>

A couple insights from the 24/25 Premier League can be seen from the Power Rating graphs:
* Liverpool's dominance, being comfortably the top offense from about GW8 onward and staying top even as many offenses fell away in the second half of the season. They are also second on defense, behind only Arsenal, who's good defensive form continued from last season.
* Of the top offenses, Bouremouth stand out as the biggest surprise, putting up good offensive numbers despite losing their top forward in Dominick Solanke over the summer.
* The three promoted teams, Southampton, Leicester, and Ipswich, are clearly the worst teams for both offense and defense. They are all very likely to be relagated, with the rest of the Premier League teams a good margin above them in terms of quality.
* On defense, Everton, Crystal Palace, and Fulham have put up good defensive numbers right up there with Manchester City.
* Nottingham Forest had been excellent defensively for a lot of the season, briefly rising to #2 in DPR, but their form has fallen off at the time of writing. They are currently 5 points clear of fifth place, but current form suggests it will be a struggle to hold on to the Championship League places.

### 3c. Upcoming Schedule Difficulty
With each team's OPR and DPR, we can look at each teams upcoming fixtures and predict number of goals scored and conceded, as well as number of clean sheets. In the Jupyter Workbook, the user can specify the fixture range to look at and a function will calculate the Opponent Difficulty Rating (ODR) and rank the teams by goals scored and also by number of clean sheets.
<figure>   
  <img src="/images/GW31-35_PredictedGoals.png" width="200" height="335">
    <figcaption><center>Predicted goals from gameweeks 31-35.</center></figcaption>
</figure>
<br/><br/>
<figure>   
  <img src="/images/GW31-35_PredictedCleanSheet.png" width="200" height="335">
    <figcaption><center>Predicted clean sheets and goals conceded from gameweeks 31-35.</center></figcaption>
</figure>

### 3d. Prediction Accuracy
Below are the linear regressions showing the accuracy of the predicted goals against each of actual goals, expected goals, and adjusted goals.
<figure>   
  <img src="/images/PredictedGoals_vs_ActualGoals.png" width="300" height="300">
    <figcaption><center>Predicted goals vs actual goals.</center></figcaption>
</figure>
<br/><br/>
<figure>   
  <img src="/images/PredictedGoals_vs_ExpectedGoals.png" width="300" height="300">
    <figcaption><center>Predicted goals vs expected goals.</center></figcaption>
</figure>
<br/><br/>
<figure>   
  <img src="/images/PredictedGoals_vs_AdjustedGoals.png" width="300" height="300">
    <figcaption><center>Predicted goals vs adjusted goals.</center></figcaption>
</figure>
<br/><br/>

This is roughly in line with other prediction approaches for soccer matches. With so few goals, soccer is inherently a high-variance sport (part of the reason for its worldwide appeal), but over many games there is certainly decent predictive capability.
<br/><br/>

The bar chart below shows the comparison between win/draw/loss percentages with the actual results over the course of the season so far.
<figure>   
  <img src="/images/EPL24_WinDrawLoss.png" width="400" height="400">
    <figcaption><center>Predicted win/draw/loss probabilities compared with actuals.</center></figcaption>
</figure>
<br/><br/>
Overall, away teams have slightly overperformed their expectations, and previous seasons' results.
<br/><br/>

Finally, the plot below compares predicted clean sheet probabilites with actuals. For clean sheet accuracy metrics, the games are grouped in increments of 10% and then the actual number of clean sheets in those games is evaluated. The plot shows the dotted line y=x, which correlates with perfect accuracy.
<figure>   
  <img src="/images/EPL_CS.png" width="400" height="400">
    <figcaption><center>Predicted clean sheet probabilities compared with actuals.</center></figcaption>
</figure>
<br/><br/>
The clean sheet probabilites are quite accuracte, especially for probabilities of 35% and below. There is an odd dip at the 45% which is out of line with the rest of the predictions, though that may be due to the small sample size.
<br/><br/>

## 4. Future Work
