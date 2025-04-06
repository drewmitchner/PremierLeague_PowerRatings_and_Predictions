import pandas as pd
import numpy as np
import math
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings

import pulp

ARS, AVL, BOU, BRE, BRI, CHE, CRY, EVE, FLH, IPS, LEI, LIV, MCI, MUN, NEW, NOT, SOU, TOT, WHU, WOL = range(20)

# global parameters
num_teams = 20 # number of teams
xG_avg = 1.45 # average number of goals per game for an average team
home_correction = 1.11 # correction applied to home team scoring more than avg and away team less
away_correction = 1.10
inflate_draw = 1.05
avgOPR = 2.19
avgDPR = 0.99
alpha_min = 0.9
alpha_decay_rate = 0.85
understat_xG_correction = 1.09 # understat seems to overpredict xG, on average
reduce_CS = 1.02
increase_xG_threshold = 1.00
increase_xG_percent = 0.55

def dec2(number, decimals=2):
    return "{:.{}f}".format(number, decimals)

class Team():
    "Stores team power rankings"
    def __init__(self, name, number, color, OPR, DPR):
        self.name = name
        self.number = number
        self.color = color
        self.OPR = OPR # offensive power ranking
        self.DPR = DPR # defensive power ranking
        self.OPR_reset = OPR # store the original rankings
        self.DPR_reset = DPR
        self.alpha = 0.1 # learning rate
        self.OPR_hist = []
        self.DPR_hist = []
        self.OPR_hist.append(OPR)
        self.DPR_hist.append(DPR)
        
    def add_team_schedule(self, master, num_teams):
        self.schedule = []
        for i in range(len(master)): # loop through all games
            if(master[i][0] == self.number): # check if this team is playing
                self.schedule.append(master[i][1]) # add their opponent
            elif(master[i][1] == self.number):
                self.schedule.append(master[i][0])
                
    def update_PR(self, xG, xGC, G, GC):
        # update the team's power rankings based on the predicted and actual goals/goals conceded
        self.OPR += (G-xG)*self.alpha # use the difference between actual goals and expected goals, with a learning rate
        self.DPR += (GC-xGC)*self.alpha
        
        self.OPR_hist.append(self.OPR)
        self.DPR_hist.append(self.DPR)
        
        if(self.alpha > alpha_min): # decay alpha over time
            self.alpha *= alpha_decay_rate
            
    def set_alpha(self, alpha_set):
        self.alpha = alpha_set
        
    def reset_PR(self):
        # reset the current power rankings to their original
        self.OPR = self.OPR_reset
        self.DPR = self.DPR_reset
        self.OPR_hist = []
        self.DPR_hist = []
        self.OPR_hist.append(self.OPR_reset)
        self.DPR_hist.append(self.DPR_reset)
     
### END Team class

## Utility functions
def reset_league_PR(teams):
    # set all teams' power rankings back to their original values
    for team in teams:
        team.reset_PR()
        
def set_leage_alpha(teams, alpha_set):
    # set the learning rate for every team in the league
    for team in teams:
        team.set_alpha(alpha_set)
        
def print_OPR_rank(teams):
    # print out teams in order of offensive power rating
    OPR_list = []
    for team in teams:
        OPR_list.append(team.OPR)
    
    enumerated_OPR = list(enumerate(OPR_list))
    sorted_OPR = sorted(enumerated_OPR, key=lambda x: x[1], reverse=True)
    sorted_teams = [index for index, _ in sorted_OPR]
    
    rank = 1
    print("Offensive Power Rankings")
    for i in sorted_teams:
        print(rank, ". ", teams[i].name, ":", dec2(teams[i].OPR))
        rank += 1
        
def print_DPR_rank(teams):
    # print out teams in order of defensive power rating
    DPR_list = []
    for team in teams:
        DPR_list.append(team.DPR)
    
    enumerated_DPR = list(enumerate(DPR_list))
    sorted_DPR = sorted(enumerated_DPR, key=lambda x: x[1])
    sorted_teams = [index for index, _ in sorted_DPR]
    
    rank = 1
    print("Defensive Power Rankings")
    for i in sorted_teams:
        print(rank, ". ", teams[i].name, ":", dec2(teams[i].DPR))
        rank += 1
        
def print_league_PR(teams):
    # print out each team's original and current power rankings
    for team in teams:
        print(team.name)
        print("OPR: ", dec2(team.OPR_reset), " -----> ", dec2(team.OPR))
        print("DPR: ", dec2(team.DPR_reset), " -----> ", dec2(team.DPR))
        print("======================")
        
def plot_season_PR(team):
    # plot one teams power rankings evolution over the course of the season
    plt.figure()
    plt.subplot(121)
    plt.plot(team.OPR_hist, linestyle='--', marker='o', color=team.color)
    plt.xlabel('GW')
    plt.ylabel('OPR')
    plt.grid(True)
    
    plt.subplot(122)
    plt.plot(team.DPR_hist, linestyle='--', marker='o', color=team.color)
    plt.xlabel('GW')
    plt.ylabel('DPR')
    plt.grid(True)
    plt.suptitle(team.name)
    
def plot_season_PR_all(teams):
    # plot one teams power rankings evolution over the course of the season
    for team in teams:
        plt.figure(1)
        plt.plot(team.OPR_hist, color=team.color)
        plt.text(len(team.OPR_hist)-1+0.05, team.OPR_hist[-1]-0.01, team.name, color=team.color)
        
        plt.figure(2)
        plt.plot(team.DPR_hist, color=team.color)
        plt.text(len(team.DPR_hist)-1+0.05, team.DPR_hist[-1]-0.01, team.name, color=team.color)
        
    plt.figure(1)
    plt.xlabel('GW')
    plt.ylabel('OPR')
    plt.title('Offensive Power Rankings')
    plt.grid(True)
    plt.rcParams["figure.figsize"] = (11.0,6.5)
    
    plt.figure(2)
    plt.xlabel('GW')
    plt.ylabel('DPR')
    plt.title('Defensive Power Rankings')
    plt.grid(True)
    plt.rcParams["figure.figsize"] = (11.0,6.5)


def expected_scores(home_team, away_team):

    home_xG = (home_team.OPR - (xG_avg - away_team.DPR))*home_correction
    away_xG = (away_team.OPR - (xG_avg - home_team.DPR))/away_correction
    
    # increase really low xG to account for observed clean sheet results
    if home_xG < increase_xG_threshold:
        home_xG = home_xG + (increase_xG_threshold - home_xG)*increase_xG_percent
        
    if away_xG < increase_xG_threshold:
        away_xG = away_xG + (increase_xG_threshold - away_xG)*increase_xG_percent
        
    return home_xG, away_xG

def poisson_distribution(xG, max_goals=6):
    distribution = []
    for k in range(max_goals+1):
        pdf = pow(xG, k)*np.exp(-xG)/np.math.factorial(k)
        if k==0:
            pdf /= reduce_CS
        distribution.append(pdf)
    
    distribution_norm = np.linalg.norm(distribution)
    return distribution/distribution_norm

def scoreline_prediction(home_team, away_team, max_goals=6):
    # returns a square matrix where rows idx represent home goals and column idx represent away goals.
    # each entry is the probability of that specific scoreline
    home_xG, away_xG = expected_scores(home_team, away_team)
    home_pdf = poisson_distribution(home_xG, max_goals)
    away_pdf = poisson_distribution(away_xG, max_goals)
    
    score_prediction = [ [0]*(max_goals+1) for i in range(max_goals+1)]
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            score_prediction[i][j] = home_pdf[i]*away_pdf[j]
            if(i == j):
                score_prediction[i][j] *= inflate_draw # inflate tie probability
    
    # Calculate the total sum
    total_sum = 0
    for row in score_prediction:
        total_sum += sum(row)
    
    # Normalize scoreline prediction
    for i in range(len(score_prediction)):
        for j in range(len(score_prediction[i])):
            score_prediction[i][j] /= total_sum
            
    return score_prediction

def result_probability(home_team, away_team):
    # first get the total scoreline matrix
    scoreline = scoreline_prediction(home_team, away_team, 6)
    
    max_goals = len(scoreline)-1
    p_home = 0
    p_tie = 0
    p_away = 0
    
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            if i==j:
                p_tie += scoreline[i][j]    
            elif i>j: # home team wins
                p_home += scoreline[i][j]
            else:
                p_away += scoreline[i][j]
    
    p_total = p_home + p_tie + p_away
    p_home = p_home/p_total
    p_tie = p_tie/p_total
    p_away = p_away/p_total
    return p_home, p_tie, p_away

def combined_xG(home_team, away_team):
    # return total number of expected goals for the game
    home_xG, away_xG = expected_scores(home_team, away_team)
    return home_xG + away_xG

def clean_sheet_odds(home_team, away_team):
    # return clean sheet odds for each team
    scoreline = scoreline_prediction(home_team, away_team)
    home_pCS = sum([row[0] for row in scoreline])
    away_pCS = sum(scoreline[0])
    
    return home_pCS, away_pCS

def run_season_PR(teams):
    log = pd.read_csv(r"data/matchlog24.csv")
    
    for i in range(0, len(log)):
        
        home_team = teams[globals()[log['home_name'][i]]]
        away_team = teams[globals()[log['away_name'][i]]]
        
        # predicted goals
        home_predG, away_predG = expected_scores(home_team, away_team)
        
        home_errG = home_predG - log['home_G'][i]
        away_errG = away_predG - log['away_G'][i]
        
        home_errxG = home_predG - log['home_xG'][i]
        away_errxG = away_predG - log['home_xG'][i]
        home_erradjG = home_predG - log['home_adjG'][i]
        away_erradjG = away_predG - log['home_adjG'][i]
        
        # record my match predictions
        p_win, p_draw, p_loss = result_probability(home_team, away_team)
            
        # record clean sheet prediction and result
        home_predCS, away_predCS = clean_sheet_odds(home_team, away_team)
        
        # get predicted result
        if(p_win > p_draw and p_win > p_loss):
            predResult = "w"
        elif(p_draw > p_loss):
            predResult = "d"
        else:
            predResult = "l"
        
        home_team.update_PR(home_predG, away_predG, log['home_adjG'][i], log['away_adjG'][i])
        away_team.update_PR(away_predG, home_predG, log['away_adjG'][i], log['home_adjG'][i])
        
        # Update match log with predictions
        log.iloc[i, log.columns.get_loc('home_predG')] = home_predG
        log.iloc[i, log.columns.get_loc('away_predG')] = away_predG
        log.iloc[i, log.columns.get_loc('home_errG')] = home_errG
        log.iloc[i, log.columns.get_loc('away_errG')] = away_errG
        log.iloc[i, log.columns.get_loc('home_errxG')] = home_errxG
        log.iloc[i, log.columns.get_loc('away_errxG')] = away_errxG
        log.iloc[i, log.columns.get_loc('home_erradjG')] = home_erradjG
        log.iloc[i, log.columns.get_loc('away_erradjG')] = away_erradjG
        log.iloc[i, log.columns.get_loc('p_win')] = p_win
        log.iloc[i, log.columns.get_loc('p_draw')] = p_draw
        log.iloc[i, log.columns.get_loc('p_loss')] = p_loss
        log.iloc[i, log.columns.get_loc('home_predCS')] = home_predCS
        log.iloc[i, log.columns.get_loc('away_predCS')] = away_predCS
        log.iloc[i, log.columns.get_loc('predResult')] = predResult
        
    # saving the matchlog dataframe
    log.to_csv('data/matchlog24.csv', index=False)

def match_summary(home_team, away_team):
    # print out match predictions for goals, clean sheet odds, winning odds
    home_predG, away_predG = expected_scores(home_team, away_team)
    p_win, p_draw, p_loss = result_probability(home_team, away_team)
    home_predCS, away_predCS = clean_sheet_odds(home_team, away_team)

    tab = [
        ["", home_team.name, away_team.name],
        ["", "home", "away"],
        ["OPR", dec2(home_team.OPR), dec2(away_team.OPR)],
        ["DPR", dec2(home_team.DPR), dec2(away_team.DPR)],
        ["Predicted Goals", dec2(home_predG), dec2(away_predG)],
        ["Clean Sheet %", dec2(home_predCS*100), dec2(away_predCS*100)],
        ["Home Win %", "Draw %", "Away Win %"],
        [dec2(p_win*100), dec2(p_draw*100), dec2(p_loss*100)]
    ]
    
    print("===========Match Summary===========")
    print(tabulate(tab))
    print("===================================")
    
    Msum = {} # return the summary stats for this match
    Msum["home_name"] = home_team.name
    Msum["away_name"] = away_team.name
    Msum["predG"] = {}
    Msum["predG"]["h"] = home_predG
    Msum["predG"]["a"] = away_predG
    Msum["predG"]["tot"] = home_predG + away_predG
    Msum["predCS"] = {}
    Msum["predCS"]["h"] = home_predCS
    Msum["predCS"]["a"] = away_predCS
    Msum["pResult"] = {}
    Msum["pResult"]["w"] = p_win
    Msum["pResult"]["d"] = p_draw
    Msum["pResult"]["l"] = p_loss
    
    return Msum

def count_less_than(numbers, threshold):
    count = 0
    for number in numbers:
        if number < threshold:
            count += 1
    return count

def calc_sG(match, understat):
    # using the shot data for the match, calculate the game-state corrected goals
    # game-state corrected goals discounts late goals scored by a team already ahead
    # and goals scored against a team playing with fewer players (red card, injury)
    match_data = understat.match(match["id"])
    roster = match_data.get_roster_data()
    current_score = [0, 0]
    home_sG = 0
    away_sG = 0
    home_times_str = []
    away_times_str = []
    home_red_str = []
    away_red_str = []
    red_discount = 0.9 # discount rate applied to goals scored against fewer players
    
    # find all the home and away goals
    for shot in match_data.get_shot_data()["h"]:
        if(shot["result"] == "Goal"):
            home_times_str.append(shot["minute"])
            
    for shot in match_data.get_shot_data()["a"]:
        if(shot["result"] == "Goal"):
            away_times_str.append(shot["minute"])
    
    # find the home and away red cards
    for player in roster["h"]:
        if(roster["h"][player]["red_card"] == "1"):
            if(roster["h"][player]["position"] == "Sub"): # if the sent off player was a sub, add both times
                sub_out = roster["h"][player]["roster_out"] # ID of the player subbed in for
                home_red_str.append(int(roster["h"][sub_out]["time"]) + int(roster["h"][player]["time"]))
            else: # just use this player's time
                home_red_str.append(int(roster["h"][player]["time"]))
                
    for player in roster["a"]:
        if(roster["a"][player]["red_card"] == "1"):
            if(roster["a"][player]["position"] == "Sub"): # if the sent off player was a sub, add both times
                sub_out = roster["a"][player]["roster_out"] # ID of the player subbed in for
                away_red_str.append(int(roster["a"][sub_out]["time"]) + int(roster["a"][player]["time"]))
            else: # just use this player's time
                away_red_str.append(int(roster["a"][player]["time"]))
    
    # cast to int
    home_times = [int(x) for x in home_times_str]
    away_times = [int(x) for x in away_times_str]
    
    home_reds = [int(x) for x in home_red_str]
    away_reds = [int(x) for x in away_red_str]
    
    home_G = len(home_times)
    away_G = len(away_times)
    
    if(home_G + away_G == 0):
        return home_sG, away_sG
    
    discount_start = 70 # minute to begin discounting
    discount_end = 90 # any goal 90 min or beyond counts 0.5 if scoring team was ahead
    sG_min = 0.5

    h_idx = 0
    a_idx = 0
    for i in range(home_G + away_G):
        # who scored?
        
        # reached max goals for home/away
        if(h_idx >= home_G):
            home_scored = 0 # must have been away goal
        elif(a_idx >= away_G):
            home_scored = 1 # must have been home goal
            
        # check the times of the next home and away goals, pick the lowest
        elif(home_times[h_idx] < away_times[a_idx]):
            home_scored = 1
        else:
            home_scored = 0
        # end who scored?
        
        # calc sG
        if(home_scored):
            num_away_reds = count_less_than(away_reds, home_times[h_idx])
            if(home_times[h_idx] <= discount_start):
                home_sG += 1.0*math.pow(red_discount, num_away_reds) # not a late goal, add the full goal
            elif(current_score[0] > current_score[1]): # check if home team is currently ahead
                
                if(home_times[h_idx] >= discount_end):
                    home_sG += sG_min*math.pow(red_discount, num_away_reds) # goals 90min or later by winning team get the minimum
                else:
                    # linear reduction every minute past 70 until 90
                    home_sG += (sG_min + (discount_end - home_times[h_idx])/(discount_end - discount_start)*(1 - sG_min)) \
                        *math.pow(red_discount, num_away_reds)
            
            else: # home team was not already winning
                home_sG += 1.0*math.pow(red_discount, num_away_reds)
            
            # increment score and idx after sG addition
            current_score[0] += 1
            h_idx += 1
            
            # end home team scored
        else:
            # start away team scored
            num_home_reds = count_less_than(home_reds, away_times[a_idx])
            if(away_times[a_idx] <= discount_start):
                away_sG += 1.0*math.pow(red_discount, num_home_reds) # not a late goal, add the full goal
            elif(current_score[1] > current_score[0]): # check if away team is currently ahead
                
                if(away_times[a_idx] >= discount_end):
                    away_sG += sG_min*math.pow(red_discount, num_home_reds) # goals 90min or later by winning team get the minimum
                else:
                    # linear reduction every minute past 70 until 90
                    away_sG += sG_min + (discount_end - away_times[a_idx])/(discount_end - discount_start)*(1 - sG_min) \
                        *math.pow(red_discount, num_home_reds)
            
            else: # home team was not already winning
                away_sG += 1.0*math.pow(red_discount, num_home_reds)
            
            # increment score and idx after sG addition
            current_score[1] += 1
            a_idx += 1
            
            # end away team scored
        # end for each goal loop, all goals accounted for
    
    return home_sG, away_sG

def calc_adjG(match, understat):
    # adjusted goals (adjG) is the unweighted average between 
    # 1. expected goals (xG)
    # 2. game-state goals (sG)
    
    home_G = float(match["goals"]["h"])
    away_G = float(match["goals"]["a"])
    
    home_xG = float(match["xG"]["h"])
    away_xG = float(match["xG"]["a"])
    
    home_sG, away_sG = calc_sG(match, understat)
    
    home_adjG = (home_xG + home_sG)/2
    away_adjG = (away_xG + away_sG)/2
    
    return home_adjG, away_adjG

def get_match_result(match):
    # takes in an understat match and returns the result (home win, draw, loss)
    
    if(match["goals"]["h"] > match["goals"]["a"]):
        return "w"
    elif(match["goals"]["h"] == match["goals"]["a"]):
        return "d"
    else:
        return "l"
    
def update_completed_matches(matches, teams, understat, reset_matchlog):
    # computes summary statistics for completed matches
    # adds them to the logged data
    
    # open matchlog
    if(reset_matchlog):
        matchlog = pd.DataFrame(columns=['id','datetime','home_name','away_name','home_predG','away_predG',
                                        'home_G','away_G','home_xG','away_xG','home_adjG','away_adjG','home_errG','away_errG',
                                        'home_errxG','away_errxG','home_erradjG','away_erradjG',
                                        'p_win','p_draw','p_loss','home_predCS','away_predCS','home_CS','away_CS',
                                        'predResult','actResult','oddsResult','odds_win','odds_draw','odds_loss'])
    else:
        matchlog = pd.read_csv(r"data/matchlog24.csv")
    
    tot_matches = 0
    for match in matches:

        if(match["isResult"] == False):
            continue # skip games that haven't been completed
            
        if(int(match['id']) in matchlog['id'].values):
            continue # we've already logged this game
        
        if(tot_matches % 20 == 0):
            print("Match #", tot_matches) # progress bar

        tot_matches += 1

        # which two teams are playing
        home_name = match["h"]["short_title"]
        home_team = teams[globals()[home_name]]
        
        away_name = match["a"]["short_title"]
        away_team = teams[globals()[away_name]]
        
        # actual goals
        home_G = float(match["goals"]["h"])
        away_G = float(match["goals"]["a"])
        # expected goals
        home_xG = float(match["xG"]["h"])/understat_xG_correction
        away_xG = float(match["xG"]["a"])/understat_xG_correction
        # adjusted goals
        home_adjG, away_adjG = calc_adjG(match, understat)
            
        if(away_G == 0):
            home_CS = 1
        else:
            home_CS = 0
                
        if(home_G == 0):
            away_CS = 1
        else:
            away_CS = 0
        
        odds_win = float(match["forecast"]["w"])
        odds_draw = float(match["forecast"]["d"])
        odds_loss = float(match["forecast"]["l"])
        
        # get odds result
        if(odds_win > odds_draw and odds_win > odds_loss):
            oddsResult = "w"
        elif(odds_draw > odds_loss):
            oddsResult = "d"
        else:
            oddsResult = "l"
            
        actResult = get_match_result(match)
        
        new_match = pd.DataFrame([{'id': match['id'], 'datetime': match['datetime'],
                                  'home_name': home_name, 'away_name': away_name,
                                  'home_predG': 0, 'away_predG': 0,
                                  'home_G': home_G, 'away_G': away_G,
                                  'home_xG': home_xG, 'away_xG': away_xG,
                                  'home_adjG': home_adjG, 'away_adjG': away_adjG,
                                  'home_errG': 0, 'away_errG': 0,
                                  'home_errxG': 0, 'away_errxG': 0,
                                  'home_erradjG': 0, 'away_erradjG': 0,
                                  'p_win': 0, 'p_draw': 0, 'p_loss': 0,
                                  'home_predCS': 0, 'away_predCS': 0,
                                  'home_CS': home_CS, 'away_CS': away_CS,
                                  'predResult': 0, 'actResult': actResult, 'oddsResult': oddsResult,
                                  'odds_win': odds_win, 'odds_draw': odds_draw, 'odds_loss': odds_loss}])
        
        matchlog = pd.concat([matchlog, new_match], ignore_index=True)
        ## END match loop
        
    # saving the matchlog dataframe
    matchlog.to_csv('data/matchlog24.csv', index=False)
    
def build_schedule(matches, teams):
    
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    
    sched = pd.DataFrame(index=np.arange(20))
    sched.insert(0, "Team", ["ARS","AVL","BOU","BRE","BRI","CHE","CRY","EVE","FLH","IPS","LEI","LIV","MCI","MUN","NEW","NOT","SOU","TOT","WHU","WOL"])
    for idx, row in matches.iterrows():
        round_no = int(row["Round Number"])
        rounds_played = len(teams[0].OPR_hist)
        #round_no = min(round_no, rounds_played)
        
        # go through each match, extract home and away, and add to sched
        sched.loc[team2int(row["Home Team"]), str(row["Round Number"]) + " Opp"] = team2int(row["Away Team"])
        sched.loc[team2int(row["Away Team"]), str(row["Round Number"]) + " Opp"] = team2int(row["Home Team"])
    
        sched.loc[team2int(row["Home Team"]), str(row["Round Number"]) + " H/A"] = "H"
        sched.loc[team2int(row["Away Team"]), str(row["Round Number"]) + " H/A"] = "A"
        
        # power rankings
        if(len(teams[team2int(row["Home Team"])].OPR_hist) <= round_no-1):
            home_OPR = teams[team2int(row["Home Team"])].OPR_hist[-1]
            home_DPR = teams[team2int(row["Home Team"])].DPR_hist[-1]
        else:
            home_OPR = teams[team2int(row["Home Team"])].OPR_hist[round_no-1]
            home_DPR = teams[team2int(row["Home Team"])].DPR_hist[round_no-1]
        
        if(len(teams[team2int(row["Away Team"])].OPR_hist) <= round_no-1):
            away_OPR = teams[team2int(row["Away Team"])].OPR_hist[-1]
            away_DPR = teams[team2int(row["Away Team"])].DPR_hist[-1]
        else:
            away_OPR = teams[team2int(row["Away Team"])].OPR_hist[round_no-1]
            away_DPR = teams[team2int(row["Away Team"])].DPR_hist[round_no-1]
    
        # predicted goals
        home_avgG = (home_OPR - (xG_avg - avgDPR))*home_correction
        away_avgG = (away_OPR - (xG_avg - avgDPR))/away_correction
        home_avgGC = (avgOPR - (xG_avg - home_DPR))/away_correction
        away_avgGC = (avgOPR - (xG_avg - away_DPR))*home_correction
        home_pG, away_pG = expected_scores(teams[team2int(row["Home Team"])], teams[team2int(row["Away Team"])])
        home_pCS, away_pCS = clean_sheet_odds(teams[team2int(row["Home Team"])], teams[team2int(row["Away Team"])])
        
        # add to schedule
        # Home
        sched.loc[team2int(row["Home Team"]), str(row["Round Number"]) + " Home OPR"] = home_OPR
        sched.loc[team2int(row["Home Team"]), str(row["Round Number"]) + " Away OPR"] = away_OPR
        sched.loc[team2int(row["Home Team"]), str(row["Round Number"]) + " Home DPR"] = home_DPR
        sched.loc[team2int(row["Home Team"]), str(row["Round Number"]) + " Away DPR"] = away_DPR
        sched.loc[team2int(row["Home Team"]), str(row["Round Number"]) + " Home avgG"] = home_avgG
        sched.loc[team2int(row["Home Team"]), str(row["Round Number"]) + " Away avgG"] = away_avgG
        sched.loc[team2int(row["Home Team"]), str(row["Round Number"]) + " Home avgGC"] = home_avgGC
        sched.loc[team2int(row["Home Team"]), str(row["Round Number"]) + " Away avgGC"] = away_avgGC
        sched.loc[team2int(row["Home Team"]), str(row["Round Number"]) + " Home pG"] = home_pG
        sched.loc[team2int(row["Home Team"]), str(row["Round Number"]) + " Away pG"] = away_pG
        sched.loc[team2int(row["Home Team"]), str(row["Round Number"]) + " Home CS"] = home_pCS
        sched.loc[team2int(row["Home Team"]), str(row["Round Number"]) + " Away CS"] = away_pCS
    
        # Away
        sched.loc[team2int(row["Away Team"]), str(row["Round Number"]) + " Home OPR"] = home_OPR
        sched.loc[team2int(row["Away Team"]), str(row["Round Number"]) + " Away OPR"] = away_OPR
        sched.loc[team2int(row["Away Team"]), str(row["Round Number"]) + " Home DPR"] = home_DPR
        sched.loc[team2int(row["Away Team"]), str(row["Round Number"]) + " Away DPR"] = away_DPR
        sched.loc[team2int(row["Away Team"]), str(row["Round Number"]) + " Home avgG"] = home_avgG
        sched.loc[team2int(row["Away Team"]), str(row["Round Number"]) + " Away avgG"] = away_avgG
        sched.loc[team2int(row["Away Team"]), str(row["Round Number"]) + " Home avgGC"] = home_avgGC
        sched.loc[team2int(row["Away Team"]), str(row["Round Number"]) + " Away avgGC"] = away_avgGC
        sched.loc[team2int(row["Away Team"]), str(row["Round Number"]) + " Home pG"] = home_pG
        sched.loc[team2int(row["Away Team"]), str(row["Round Number"]) + " Away pG"] = away_pG
        sched.loc[team2int(row["Away Team"]), str(row["Round Number"]) + " Home CS"] = home_pCS
        sched.loc[team2int(row["Away Team"]), str(row["Round Number"]) + " Away CS"] = away_pCS

    ## END loop through matches
    return sched

def team2int(string_in):
    # convert the string in schedule to the team number
    match string_in:
        case "Arsenal":
            return ARS
        case "Aston Villa":
            return AVL
        case "Bournemouth":
            return BOU
        case "Brentford":
            return BRE
        case "Brighton":
            return BRI
        case "Chelsea":
            return CHE
        case "Crystal Palace":
            return CRY
        case "Everton":
            return EVE
        case "Fulham":
            return FLH
        case "Ipswich":
            return IPS
        case "Leicester":
            return LEI
        case "Liverpool":
            return LIV
        case "Man City":
            return MCI
        case "Man Utd":
            return MUN
        case "Newcastle":
            return NEW
        case "Nott'm Forest":
            return NOT
        case "Southampton":
            return SOU
        case "Spurs":
            return TOT
        case "West Ham":
            return WHU
        case "Wolves":
            return WOL
        
def fixture_difficulty_rank(schedule, teams, num_rounds, start_round=1, discount_rate=1):
    # assess offensive and defensive difficulty ranks over the next couple gameweeks
    rounds = range(start_round, num_rounds+start_round)
    print('GW Range: ' +str(start_round)+'-'+str(num_rounds+start_round-1))
    if (rounds[-1] < 38):
        sub_sched = schedule.loc[:, str(min(rounds)) + ' Opp':str(rounds[-1]+1) + ' Opp']
    else:
        sub_sched = schedule.loc[:, str(min(rounds)) + ' Opp':]
    
    # Opponent metrics
    OPR_opp_list = [0 for i in range(num_teams)]
    DPR_opp_list = [0 for i in range(num_teams)]
    avgG_opp_list = [0 for i in range(num_teams)]
    avgGC_opp_list = [0 for i in range(num_teams)]
    # Own team metrics
    pG_list = [0 for i in range(num_teams)]
    pGC_list = [0 for i in range(num_teams)]
    pCS_list = [0 for i in range(num_teams)]
    
    # Loop through all the teams
    for team in range(num_teams):
        for gw in rounds: # Loop through all the desired gameweeks
            home_away = sub_sched.loc[team][str(gw) + " H/A"]

            if(home_away == "H"):
                # Opponent's metrics
                OPR_opp_list[team] += sub_sched.loc[team][str(gw) + " Away OPR"]
                DPR_opp_list[team] += sub_sched.loc[team][str(gw) + " Away DPR"]
                avgG_opp_list[team] += sub_sched.loc[team][str(gw) + " Away avgG"]
                avgGC_opp_list[team] += sub_sched.loc[team][str(gw) + " Away avgGC"]
                # Own team's predictions
                pG_list[team] += sub_sched.loc[team][str(gw) + " Home pG"]
                pGC_list[team] += sub_sched.loc[team][str(gw) + " Away pG"]
                pCS_list[team] += sub_sched.loc[team][str(gw) + " Home CS"]
            else:
                # Opponent's metrics
                OPR_opp_list[team] += sub_sched.loc[team][str(gw) + " Home OPR"]
                DPR_opp_list[team] += sub_sched.loc[team][str(gw) + " Home DPR"]
                avgG_opp_list[team] += sub_sched.loc[team][str(gw) + " Home avgG"]
                avgGC_opp_list[team] += sub_sched.loc[team][str(gw) + " Home avgGC"]
                # Own team's predictions
                pG_list[team] += sub_sched.loc[team][str(gw) + " Away pG"]
                pGC_list[team] += sub_sched.loc[team][str(gw) + " Home pG"]
                pCS_list[team] += sub_sched.loc[team][str(gw) + " Away CS"]
                
        # END gameweek loop
    # END team loop
    
    # Get Offensive difficulty ranking (ODR)
    enumerated_pG = list(enumerate([x for x in pG_list]))
    sorted_pG = sorted(enumerated_pG, key=lambda x: x[1], reverse=True)
    sorted_teams = [index for index, _ in sorted_pG]
    sorted_pG = [x[1] for x in sorted_pG]
    
    sorted_names = [0 for i in range(num_teams)]
    for i in range(num_teams):
        sorted_names[i] = teams[sorted_teams[i]].name
    
    data = {'Team': sorted_names,
            'predG': sorted_pG,
            'ODR': [avgGC_opp_list[x]/num_rounds for x in sorted_teams]}
    ODR_df = pd.DataFrame(data)
    
    display(ODR_df)
    
    # get defensive difficulty ranking (DDR)
    enumerated_pCS = list(enumerate([x for x in pCS_list]))
    sorted_pCS = sorted(enumerated_pCS, key=lambda x: x[1], reverse=True)
    sorted_teams = [index for index, _ in sorted_pCS]
    sorted_pCS = [x[1] for x in sorted_pCS]
    
    sorted_names = [0 for i in range(num_teams)]
    for i in range(num_teams):
        sorted_names[i] = teams[sorted_teams[i]].name
    
    data = {'Team': sorted_names,
            'predCS': sorted_pCS,
            'predGC': [pGC_list[x] for x in sorted_teams],
            'DDR': [avgG_opp_list[x]/num_rounds for x in sorted_teams]}
    DDR_df = pd.DataFrame(data)
    
    display(DDR_df)
    return ODR_df, DDR_df

def get_next_GW_fixtures(matches, num_matches=10):
    # returns the next gameweek's fixture list, or user input number of matches
    counter = 0
    GW_matches = []
    
    for match in matches:
        if(match["isResult"] == True):
            continue # skip games that have been completed
        
        # add the match to the GW's matches
        GW_matches.append(match)
        counter += 1
        
        if(counter==num_matches):
            break
        
        
    # END loop through all matches
    
    return GW_matches

def GW_predG_rank(Msums):
    # print a rank of which teams are likely to score the most goals in the upcoming gameweek
    
    names = []
    predG = []
    for Msum in Msums:
        names.append(Msum["home_name"])
        names.append(Msum["away_name"])
        
        predG.append(Msum["predG"]["h"])
        predG.append(Msum["predG"]["a"])
        
    enumerated_predG = list(enumerate(predG))
    sorted_predG = sorted(enumerated_predG, key=lambda x: x[1], reverse=True)
    sorted_teams = [index for index, _ in sorted_predG]
    
    rank = 1
    print("GW Predicted Goals")
    for i in sorted_teams:
        print(rank, ". ", names[i], ":", dec2(predG[i]))
        rank += 1
        
    print("=============================")
    
def GW_predCS_rank(Msums):
    # print a rank of which teams are likeliest to keep a clean sheet in this gameweek
    
    names = []
    CS_list = []
    for Msum in Msums:
        names.append(Msum["home_name"])
        names.append(Msum["away_name"])
        
        CS_list.append(Msum["predCS"]["h"])
        CS_list.append(Msum["predCS"]["a"])
        
    enumerated_CS = list(enumerate(CS_list))
    sorted_CS = sorted(enumerated_CS, key=lambda x: x[1], reverse=True)
    sorted_teams = [index for index, _ in sorted_CS]
    
    rank = 1
    print("GW Predicted Clean Sheet %")
    for i in sorted_teams:
        print(rank, ". ", names[i], ":", dec2(CS_list[i]*100))
        rank += 1
        
    print("=============================")
    
def GW_totalG_rank(Msums):
    # print a rank of the matches likeliest to have the highest number of total goals in this gameweek
    matchups = []
    predG_tot = []
    
    for Msum in Msums:
        matchups.append(Msum["home_name"] + " vs " + Msum["away_name"])
        
        predG_tot.append(Msum["predG"]["tot"])
        
    enumerated_totG = list(enumerate(predG_tot))
    sorted_totG = sorted(enumerated_totG, key=lambda x: x[1], reverse=True)
    sorted_matchups = [index for index, _ in sorted_totG]
    
    rank = 1
    print("GW Total Goals")
    for i in sorted_matchups:
        print(rank, ". ", matchups[i], ":", dec2(predG_tot[i]))
        rank += 1
        
    print("=============================")
    
def GW_summary(matches, teams, num_matches=10):
    # print the match summary for all matches in the given range
    
    GW_matches = get_next_GW_fixtures(matches, num_matches)
    M_sums = [] # list of match summaries
    
    for match in GW_matches:
        # which two teams are playing
        home_name = match["h"]["short_title"]
        home_team = teams[globals()[home_name]]
        
        away_name = match["a"]["short_title"]
        away_team = teams[globals()[away_name]]
        
        # print match summary
        Msum = match_summary(home_team, away_team)
        M_sums.append(Msum)
    
    # END loop through GW
    
    # summary statistics across GW
    GW_predG_rank(M_sums)
    GW_predCS_rank(M_sums)
    GW_totalG_rank(M_sums)

# add fixture strength adjustments to expected score each gameweek
def add_gw_ppoints(p_df, sched, num_gw = 5, start_gw = 1, decay_rate=1):
    
    for gw in range(start_gw, start_gw + num_gw + 1): # loop through each gw
        for i in range(0, len(p_df)): # loop through all the players
            # get the team and position
            current_team = p_df.loc[i, "team"]
            current_pos = p_df.loc[i, "position"]
            
            # team prediction
            home_away = sched.loc[current_team, str(gw) + " H/A"]
            
            if home_away == "H":
                if current_pos == "GK" or current_pos == "DEF":
                    DPR_1 = sched.loc[current_team, str(1) + " Home DPR"]
                    DPR_N = sched.loc[current_team, str(gw) + " Home DPR"]
                    fixture_adj = (sched.loc[current_team, str(gw) + " Home avgGC"]/home_correction)/\
                        max(sched.loc[current_team, str(gw) + " Away pG"], 0.3)*\
                        (DPR_1/DPR_N)**1
                    fixture_adj = min(fixture_adj, 1.5)
                else: # attacking player
                    OPR_1 = sched.loc[current_team, str(1) + " Home OPR"]
                    OPR_N = sched.loc[current_team, str(gw) + " Home OPR"]
                    fixture_adj = sched.loc[current_team, str(gw) + " Home pG"]/\
                        (sched.loc[current_team, str(gw) + " Home avgG"]/home_correction)*\
                        (OPR_N/OPR_1)*1
            else:
                if current_pos == "GK" or current_pos == "DEF":
                    DPR_1 = sched.loc[current_team, str(1) + " Away DPR"]
                    DPR_N = sched.loc[current_team, str(gw) + " Away DPR"]
                    fixture_adj = (sched.loc[current_team, str(gw) + " Away avgGC"]*away_correction)/\
                        max(sched.loc[current_team, str(gw) + " Home pG"], 0.3)*\
                        (DPR_1/DPR_N)**1
                    fixture_adj = min(fixture_adj, 1.5)
                else: # attacking player
                    OPR_1 = sched.loc[current_team, str(1) + " Away OPR"]
                    OPR_N = sched.loc[current_team, str(gw) + " Away OPR"]
                    fixture_adj = sched.loc[current_team, str(gw) + " Away pG"]/\
                        (sched.loc[current_team, str(gw) + " Away avgG"]*away_correction)*\
                        (OPR_N/OPR_1)**1
            
            p_df.loc[i, str(gw) + " pPoints"] = (p_df.loc[i, "points_per_game"] - 2)*fixture_adj + 2
            p_df.loc[i, str(start_gw) +"-" + str(gw) + " pPoints"] = sum(p_df.loc[i, str(x) + " pPoints"]*decay_rate**(x-start_gw) for x in range(start_gw, gw+1))



            
def LP_Optimize_Team(players, opt_gw, budget=825, include_players=[], exclude_players=[]):
    # set up linear programming optimization

    ## Set variables
    x = pulp.LpVariable.dict("player", range(0, len(players)), cat=pulp.LpBinary)
    y = pulp.LpVariable.dict("captain", range(0, len(players)), cat=pulp.LpBinary)

    # set up objective function
    prob = pulp.LpProblem("FantasyFootball", pulp.LpMaximize)
    prob += pulp.lpSum(players.loc[i, opt_gw + " pPoints"] * (x[i] + y[i]) for i in range(0, len(players)))

    ## Set constraints
    # Max 11 players
    prob += sum(x[i] for i in range(0, len(players))) ==  11

    # One goalkeeper
    prob  += sum(x[i] for i in range(0, len(players)) if players["position"][i] == "GK") == 1

    ## Three to Five Defenders
    prob  += sum(x[i] for i in range(0, len(players)) if players["position"][i] == "DEF") >= 3
    prob  += sum(x[i] for i in range(0, len(players)) if players["position"][i] == "DEF") <= 5
 
    ## Three to Five Midfielders
    prob  += sum(x[i] for i in range(0, len(players)) if players["position"][i] == "MID") >= 3
    prob  += sum(x[i] for i in range(0, len(players)) if players["position"][i] == "MID") <= 5
 
    ## One to Three Attackers
    prob  += sum(x[i] for i in range(0, len(players)) if players["position"][i] == "FWD") >= 1
    prob  += sum(x[i] for i in range(0, len(players)) if players["position"][i] == "FWD") <= 3

    ## Max 3 players from any team
    for t in range(0, 20):
        prob  += sum(x[i] for i in range(0, len(players)) if players["team"][i] == t) <= 3

    ## One captain
    prob  += sum(y[i] for i in range(0, len(players))) == 1

    for i in range(len(players)):  # captain must also be on team
        prob += (x[i] - y[i]) >= 0

    # 100MM budget (100mil in dataset is 1000)
    # This does not account for subs however
    # Decrease budget by the price of the cheapest player in each pos
    # 4 for GK, 4 for DEF, 4.5 for MID, 5.0 for FW - 17.5 total
    prob += sum(x[i] * players["now_cost"][i] for i in range(0, len(players))) <= budget  # total cost

    # Optional: force player picks
    for in_player in include_players:
        prob  += sum(x[i] for i in range(0, len(players)) if players["web_name"][i] == in_player) == 1
    
    for out_player in exclude_players:
        prob  += sum(x[i] for i in range(0, len(players)) if players["web_name"][i] == out_player) == 0
    
    print(prob.solve()) # Outputs 1 if successful
    print(dec2(pulp.value(prob.objective))) # Show points total
    
    # print optimal team
    for i in range(0, len(players)):
        if pulp.value(x[i]) == 1:
            if pulp.value(y[i]) == 1:
                print("{player} (C):   \t{points} points".format(player = players["web_name"][i],
                                                             points = dec2(players[opt_gw + " pPoints"][i]*2)))
            else:
                print("{player}:   \t{points} points".format(player = players["web_name"][i],
                                                         points = dec2(players[opt_gw + " pPoints"][i])))