from typing import Tuple
import numpy as np
import itertools
from numpy.core.fromnumeric import shape
from math import factorial

class Optimiser:

    def __init__(self, fixtures, batch:int, start_point:int):
        self.fixtures = fixtures # get the fixtures
        self.batch = batch # represents the number of teams in a selection
        self.start_point = start_point # represents the fixture date to start the calculation from; should be greater than atleast 3 to ensure enough games for each team has passed
    
    def invalid_key_error(self) -> None:
        raise Exception('Invalid side or team specified')

    def get_permutations(self) -> tuple:
        dates = self.fixtures['Date'] # extract the dates
        dates.drop_duplicates(keep='first', inplace=True) #drop  duplicates
        dates_ = dates.tolist() # convert to list

        reverse = list(reversed(dates_)) # reverse the dates with the oldest being first and latest the latter
        arr_perms = []
        selected_dates = [] # dates referring to that of the selected fixtures
        for _, dt in enumerate(reverse):
            if _ >= self.start_point:
                df = self.fixtures.loc[self.fixtures['Date']==dt] # get the fixtures for the selected date
                if len(df) >= self.batch:
                    # get all possible combinations
                    perms = list(itertools.permutations(df['Home'], self.batch))
                    # create a new empty list to store unique combination sets
                    sorted_perms = {} 
                    result = {}
                    for itr, val in enumerate(perms):
                    
                        char = "".join(val) # combine all elements of the set into a string
                        char = char.replace(" ", "") # remove the empty spaces
                        # arrange alphabetically
                        char = sorted(char)
                        # combine the elements above into a single string again
                        char = "".join(char)
                        char = str(char.lower())

                        sorted_perms[char] = val
                    
                    for key, value in sorted_perms.items(): # create a new dict to filter out duplicate keys from the above dict
                        if key not in result.keys():
                            result[key] = value
                        
                    arr_perms.append(result)
                    selected_dates.append(dt)
        
        return (arr_perms, selected_dates)

    def league_average(self) -> tuple:
        # calculate league averages
        league_avg_home = self.fixtures['Home Goals'].mean()
        league_avg_away = self.fixtures['Away Goals'].mean()
        
        return (league_avg_home, league_avg_away)
    
    def home_average(self, home:str) -> tuple:
        home_team_home_games = self.fixtures.loc[(self.fixtures['Home']==home)] # get all home games for the home team
        home_team_away_games = self.fixtures.loc[(self.fixtures['Away']==home)] # get all away games for the home team

        home_goal_average = home_team_home_games['Home Goals'].mean() # get the average home goals per home match
        home_allowed_average = home_team_home_games['Away Goals'].mean() # get the average allowed goals per home match

        return (home_goal_average, home_allowed_average)

    def away_average(self, away:str) -> tuple:
        away_games = self.fixtures.loc[(self.fixtures['Away'] == away)] # get all away games
        away_goal_average = away_games['Away Goals'].mean() # get the average goals score by the away side on its away matches
        away_allowed_average = away_games['Home Goals'].mean() # get the average 
        total_away_team_goal_average = (away_goal_average + away_allowed_average)/2

        return (away_goal_average, away_allowed_average)

    def attack_strength(self, side:str, team:str) -> float:
        if side == 'Away':
            attack_strength = self.away_average(team)[0]/self.league_average()[1]
        elif side == 'Home':
            attack_strength =  self.home_average(team)[0]/self.league_average()[0]
        else:
            self.invalid_key_error()

        return attack_strength           

    def defense_strength(self, side:str, team:str) -> float:
        if side == 'Away':
            defensive_strength = self.away_average(team)[1]/self.league_average()[1]
        elif side == 'Home':
            defensive_strength =  self.home_average(team)[1]/self.league_average()[0]
        else:
            self.invalid_key_error()

        return defensive_strength        

    def goal_expectancy(self, sides:dict, side:str) -> float:
        if side == 'Home':
            goal_expectancy = self.attack_strength(side, sides[side]) * self.defense_strength('Away', sides['Away']) * self.league_average()[0]
        elif side == 'Away':
            goal_expectancy = self.attack_strength(side, sides[side]) * self.defense_strength('Home', sides['Home']) * self.league_average()[1]
        else:
            self.invalid_key_error()

        return goal_expectancy
    
    def outcome_probability(self, sides:dict) -> tuple:
        goals = [0,1,2,3,4,5]
        home_probability = []
        away_probability = []
        home_probability += [(self.goal_expectancy(sides, 'Home')**goal  * 2.71828**(-self.goal_expectancy(sides, 'Home')))/factorial(goal) for goal in goals]
        away_probability += [(self.goal_expectancy(sides, 'Away')**goal  * 2.71828**(-self.goal_expectancy(sides, 'Away')))/factorial(goal) for goal in goals]

        matrix = np.zeros((len(goals), len(goals)))
        for i in range(len(matrix)):
            matrix[i][:] = np.round(home_probability[i] * np.array(away_probability),2)

        # total draw probability
        draw = np.diagonal(matrix).sum()

        # total loss probability
        loss = sum(np.diagonal(matrix, x).sum() for x in [1,2,3,4])
        # total win probability
        win = sum(np.diagonal(matrix, _x).sum() for _x in [-1,-2,-3,-4])

        return (matrix, win, draw, loss)
    
    def get_team_combinations(self, selection:dict):
        home_sides = selection.get('Home')
        away_sides = selection.get('Away')

        perms = list(itertools.permutations(home_sides, self.batch)) # gets all possible combinations
        sorted_perms = {}
        result = {}
        for _, val in enumerate(perms):
            char = "".join(val)
            char = char.replace(" ", "")
            char = sorted(char)
            char = "".join(char)
            char = str(char.lower())
            sorted_perms[char] = val

        for key, val in sorted_perms.items():
            if key not in result.keys():
                result[key] = val
        
        return result
    
    def get_best_fit(self, selection:dict) -> dict:
        combinations = self.get_team_combinations(selection)
        outcomes = {}
   
        for k_, teams in combinations.items(): # loop through the combinations
            win_probability = []
            for t_ in teams: # loop through the teams in each combination
                index_ = selection['Home'].index(t_) # get the index of the ith team
                away_team = selection['Away'][index_] # get the away team matched with the ith team

                # get probabilities for the match up
                matrix, win, draw, loss = self.outcome_probability({'Home':t_, 'Away':away_team})
                if (win - draw) > 0.6:
                    win_probability.append(win)

            outcomes[k_] = {'Selection':teams, 'win':np.product(win_probability)}

        return outcomes
           

            

            


        
