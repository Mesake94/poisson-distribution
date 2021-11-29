from datetime import date
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
import dateparser
from math import factorial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as ticker
import streamlit as st
import optimiser as op


st.title('Football Sports Betting (European Leagues)')

st.sidebar.title('League Selection')
sidebar = st.sidebar

league_shortcut = {
        'Premier League':['premier-league-table','premier-league-results'], 'Championship':['championship-table','championship-results'],
        'Seria A':['seria-a-table','seria-a-results'], 'Ligue 1':['ligue-1-table', 'ligue-1-results'],
        'Bundesliga':['bundesliga-table','bundesliga-results'], 'La Liga':['la-liga-table','la-liga-results'],
        'League One':['league-one-table','league-one-results'], 'League Two':['league-two-table','league-two-results'],
        'Scottish Premiership':['scottish-premiership-table','scottish-premiership-results'],
        'Scottish Championship':['scottish-championship-table','scottish-championship-results'],
        'Scottish League One':['scottish-league-one-table','scottish-league-one-results'],
        'Scottish League Two':['scottish-league-two-table','scottish-league-two-results']
    }
# FUNCTIONS TO CACHE
@st.cache(suppress_st_warning=True)
def load_dataset(league, season, league_shortcut): # function to load league table from skyports website
    standings = None
    season_ = season.split("/")[0] # get the season
    league_table = requests.get(f"https://www.skysports.com/{league_shortcut[league][0]}/{season_}")
    if league_table:
        soup = BeautifulSoup(league_table.content, 'html.parser')
        if soup:
            rows = soup.find_all('tr') # get the table rows
            standings = []
            for index, row in enumerate(rows):
                if index > 0: # skip the header row
                    columns = row.find_all('td') # get all the columns
                    position = int(columns[0].text)
                    team = columns[1].text.split("\n")[1]
                    played = int(columns[2].text)
                    wins = int(columns[3].text)
                    draws = int(columns[4].text)
                    loss = int(columns[5].text)
                    goals_for = int(columns[6].text)
                    goals_against = int(columns[7].text)
                    goal_difference = int(columns[8].text)
                    points = int(columns[9].text)
                    standings.append({
                        'Position':position, 'Team':team, 'Played':played,
                        'W':wins, 'D':draws, 'L':loss,
                        'GF':goals_for, 'GA':goals_against,
                        'GD':goal_difference, 'Pts':points
                    })
    if standings:
        return pd.DataFrame.from_dict(standings)
    
@st.cache(suppress_st_warning=True)
def load_fixture_results(league, season, league_shortcut): # function to load league fixture results 
    
    season__ = season.split("/") # split to change the slash with the hyphen
    season__ = season__[0]+"-"+season__[1]
    try:
        r = requests.get(f"https://www.skysports.com/{league_shortcut[league][1]}/{season__}")
        if r.content:
            soup2 = BeautifulSoup(r.content, 'html.parser')
    except Exception as e:
        print(e)

    main_ = soup2.find("div", class_="fixres__body")

    dates = []
    prefix = ['st', 'nd', 'rd', 'th']
    year = season__.split("-")[0]

    # extract the dates the matches were played
    for date in main_.find_all('h4'):
        date = date.text.split(" ")[1:]
        for _ in prefix:
            if _ in date[0]:
                date[0] = date[0].split(_)[0]
        date.append(year)
        dates.append(dateparser.parse(f"{date[0]}-{date[1]}-{date[2]}").date())

    parent_div = main_.find_all('div', class_="fixres__item")
    fixtures = []
    if parent_div:
        index = 0
        for itr, div in enumerate(parent_div): # loop through the parent div
            a_ = div.a # get the next link tag element inside the parent div
            if a_:
                # first get the sides
                side1 = a_.find('span', class_="matches__item-col matches__participant matches__participant--side1").text.split("\n")[2]
                side2 = a_.find('span', class_="matches__item-col matches__participant matches__participant--side2").text.split("\n")[2]
                # second get the scores
                span_ = a_.find_all('span', class_="matches__teamscores-side")

                score1 = span_[0].text.split("\n")[1]
                score1 = int(score1.split(" ")[16])

                score2 = span_[1].text.split("\n")[1]
                score2 = int(score2.split(" ")[16])
                previous = div.fetchPrevious()[0]

                if previous.get('class')[0] == "fixres__header2" and len(fixtures) > 1:
                    index += 1

                fixtures.append({'Home':side1, 'Away':side2, 'Home Goals':score1, 'Away Goals':score2, 'Date':dates[index]})

    if fixtures:
        return pd.DataFrame.from_dict(fixtures)

@st.cache(suppress_st_warning=True)
def calculate_odds(home, away, fixture_df): # function to calculate probabilities of the diffent possible scorelines
    # calculate league averages first
    league_avg_home = fixture_df['Home Goals'].mean()
    league_avg_away = fixture_df['Away Goals'].mean()
    total_league_avg = (league_avg_home + league_avg_away)/2

    # calculate averages for the home side
    home_team_home_games = fixture_df.loc[(fixture_df['Home'] == home)] # get all home games
    home_team_away_games = fixture_df.loc[(fixture_df['Away'] == home)]
    home_goal_average = home_team_home_games['Home Goals'].mean() # get the average home goals per home match
    home_allowed_average = home_team_home_games['Away Goals'].mean() # get the average allowed goals per home match
    total_home_team_goal_average = (home_goal_average + home_allowed_average)/2

    # calculate averages for the away side
    away_games = fixture_df.loc[(fixture_df['Away'] == away)] # get all away games
    away_goal_average = away_games['Away Goals'].mean() # get the average goals score by the away side on its away matches
    away_allowed_average = away_games['Home Goals'].mean() # get the average 
    total_away_team_goal_average = (away_goal_average + away_allowed_average)/2

    # calculation of the attacking strength for both sides
    home_attack_strength = home_goal_average/league_avg_home
    away_attack_strength = away_goal_average/league_avg_away

    # calcuation of the defensive strengh for both sides
    home_defence_strength = home_allowed_average/league_avg_home
    away_defence_strength = away_allowed_average/league_avg_away

    # calculation of goal expectancy for both sides
    home_team_goal_expectancy = home_attack_strength * away_defence_strength * league_avg_home
    away_team_goal_expectancy = away_attack_strength * home_defence_strength * league_avg_away

    # POISSON DISTRIBUTION
    goals = [0,1,2,3,4,5]
    home_probability = []
    away_probability = []

    home_probability += [(home_team_goal_expectancy**goal  * 2.71828**(-home_team_goal_expectancy))/factorial(goal) for goal in goals]
    away_probability += [(away_team_goal_expectancy**goal  * 2.71828**(-away_team_goal_expectancy))/factorial(goal) for goal in goals]
    matrix = np.zeros((len(goals),len(goals)))
    for i in range(len(matrix)):
        matrix[i][:] = np.round(home_probability[i] * np.array(away_probability),2)

    # total draw probability
    draw = np.diagonal(matrix).sum()

    # total loss probability
    loss = sum(np.diagonal(matrix, x).sum() for x in [1,2,3,4])
    # total win probability
    win = sum(np.diagonal(matrix, _x).sum() for _x in [-1,-2,-3,-4])
    return (matrix, home_team_home_games, away_games, home_attack_strength, home_defence_strength, away_attack_strength, away_defence_strength, home_team_goal_expectancy, away_team_goal_expectancy, win, draw, loss)

@st.cache(suppress_st_warning=True)
def backtest(df, batch, max_draw, min_win): # function to back test results and evaluate performance of the poisson distribution
    # print(df)
    dates = df['Date'] # extract the dates
    dates.drop_duplicates(keep='first', inplace=True) #drop  duplicates
    dates_ = dates.tolist() # convert to list
    # create a dataframe to store the predicted and actual results of the selected fixtures
    results = []

    reverse = list(reversed(dates_)) # reverse the dates with the oldest being first and latest the latter
    for _, dt in enumerate(reverse):
        if _ >= 8:
            fixture_df = df.loc[df['Date']==dt] # filter the dataframe to get all rows based on the date
            historical_df = df.loc[df['Date'] < dt] # get rows with dates prior to the specified date
            if len(fixture_df) >= 3: # check for atleast 3 fixtures for the date
                home_teams = fixture_df['Home'].tolist() # get the home teams from the fixture
                away_teams = fixture_df['Away'].tolist() # get the away teams from the fixture
                total_win, total_draw, total_loss, selected_home, selected_away = accumulator(home_teams, away_teams, historical_df, batch, max_draw=max_draw, min_win=min_win)

                predictions = []
                for itr, team in enumerate(selected_home): # loop through the selected home teams
                    row = fixture_df.loc[df['Home']==team] # get the row for the corresponding home team
                    home_goals = row.loc[row.index, 'Home Goals'].values # get the home goals
                    away_goals = row.loc[row.index, 'Away Goals'].values # get the away goals
                    if home_goals[0] >  away_goals[0]: # check if the home team won
                        predicted = 'Win'
                    elif home_goals[0] < away_goals[0]: # check if the home team lost
                        predicted = 'Lost'
                    else:
                        predicted = 'Draw' 

                    predictions.append(predicted)
                
                if 'Lost' in predictions or 'Draw' in predictions:
                    actual = 'Lost'
                else:
                    actual = 'Win'
                
                results.append({
                    'Home':f"{selected_home}", 'Away':f"{selected_away}",
                    'Predicted':'Win', 'Actual':actual
                })
                
    return pd.DataFrame.from_records(results)


@st.cache(suppress_st_warning=True)
def accumulator(home_teams, away_teams, fixtures, batch, *args, **kwargs): # function to calculate win, draw and lose probability for a selection of teams
    # the batch refers to the number of teams to include in a selection group for the accumulator bet
    if len(home_teams) != len(away_teams) and len(home_teams) >= batch:
        st.warning('The number of home and away teams do not match!')
        return False

    max_draw = kwargs['max_draw'] if 'max_draw' in kwargs.keys() else 0.1 # get the max draw ratio
    min_win = kwargs['min_win'] if 'min_win' in kwargs.keys() else 0.6 # get the min win ratio
    
    wins = []
    draws = []
    losses = []
    selected_home = []
    selected_away = []
    for x, y in enumerate(home_teams):
        matrix, home_team_home_games, away_games, home_attack_strength, home_defence_strength, away_attack_strength, away_defence_strength, home_team_goal_expectancy, away_team_goal_expectancy, win, draw, loss = calculate_odds(home_teams[x], away_teams[x], fixtures)
        if wins and len(wins) >= batch: # check if length of wins list is greater than 3 (batch size)
            min_val = min(wins) # get the minimum value of the list
            min_index = wins.index(min_val) # get the index of the minimum value
            # check if the draw probability is less than the max value of the draws list
            # max_draw = max(draws) # get the maximum value of draw from the list
            # max_draw_index = draws.index(max_draw) # get the index of the max draw value
            if win > min_val: # check if the win prob is greater than the min value of the list
                print(f'win: {win} draw: {draw} loss: {loss}')
                # pop the minimum win and maximum draw from the list
                wins.pop(min_index) # remove the minimum value
                draws.pop(min_index) # remove the maxmimum draw
                selected_home.pop(min_index) # remove the home side from the selected home list
                selected_away.pop(min_index) # remove the corresponding away side from the selected away list
                
                # update the win and draw list with new values
                wins.append(win)
                draws.append(draw)
                selected_home.append(y)
                selected_away.append(away_teams[x])
        elif win >= min_win and draw < max_draw:
            wins.append(win)
            draws.append(draw)
            losses.append(loss)
            selected_home.append(y)
            selected_away.append(away_teams[x])

    total_win = np.prod(wins)
    total_draw = np.prod(draws)
    total_loss = np.prod(losses)

    return (total_win, total_draw, total_loss, selected_home, selected_away)

def render_UI():
    # renders sidebar element
    sidebar = st.sidebar
    
    with sidebar:
        league = sidebar.selectbox('Select League',
            options=(
                'Premier League',
                'Championship',
                'Seria A',
                'Ligue 1',
                'Bundesliga',
                'La Liga',
                'League One',
                'League Two',
                'Scottish Premiership',
                'Scottish Championship',
                'Scottish League One',
                'Scottish League Two'
            )
        )
        season = sidebar.selectbox('Select Season',
            options=(
                '2021/22',
                '2020/21',
                '2019/20'
            )
        )
    
    if league or season:
        standings = load_dataset(league, season, league_shortcut)
        st.subheader(f'{league} Table - {season}')
        standing_table = st.dataframe(standings, 700, 700)
        fixtures = load_fixture_results(league, season, league_shortcut)
    
    sidebar.subheader('Team Selection')
    
    single = sidebar.checkbox('Single', key='single')
    multiple = sidebar.checkbox('Multiple', key='multiple')
    if single == True and multiple == False:
        home = sidebar.selectbox('Home Side',
            options=standings['Team']
        )
        away = sidebar.selectbox('Away Side',
            options=standings['Team']
        )

        calculate = sidebar.button('Predict Possible Outcomes', key='calculate')

        if calculate and home == away:
            st.error('Please choose different teams')
        elif calculate:
            matrix, home_team_home_games, away_games, home_attack_strength, home_defence_strength, away_attack_strength, away_defence_strength, home_team_goal_expectancy, away_team_goal_expectancy, win, draw, loss = calculate_odds(home, away, fixtures)
            if len(matrix)> 0:
                st.success('Probabilities of possible outcomes successfully calculated.')
                st.subheader('Home Team Games')
                st.dataframe(home_team_home_games ,700, 600)
                st.subheader('Away Team games')
                st.dataframe(away_games, 700, 600)

                st.subheader('General Statistics')
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric('Home Attack Strength', value=str(home_attack_strength))
                with col2:
                    st.metric('Home Defensive Strength', value=str(home_defence_strength))
                with col3:
                    st.metric('Away Attack Strength', value=str(away_attack_strength))
                with col4:
                    st.metric('Away Defensive Strength', value=str(away_defence_strength))
                with col5:
                    st.metric('Home Team Goal Expectancy', value=str(home_team_goal_expectancy))
                with col6:
                    st.metric('Away Team Goal Expectancy', value=str(away_team_goal_expectancy))

                # render the plot
                st.subheader('Probabilities of Possible Scores')
                fig = plt.figure()
                fig, ax = plt.subplots(1,1, figsize=(8,8))
                heatplot = ax.imshow(matrix, cmap='plasma_r')
                # ax.set_xticklabels(np.array(away_probability))
                # ax.set_yticklabels(flight_matrix.index)
                for (j,i),label in np.ndenumerate(matrix):
                    ax.text(i,j,label,ha='center',va='center')
                    ax.text(i,j,label,ha='center',va='center')
                    
                cbar = fig.colorbar(heatplot)

                ax.set_xlabel(f"Away Goals - {away}")
                ax.set_ylabel(f"Home Goals - {home}")

                st.pyplot(fig)
                win_col, draw_col, loss_col = st.columns(3)

                with win_col:
                    st.metric('Win', value=str(win))
                with draw_col:
                    st.metric('Draw', value=str(draw))
                with loss_col:
                    st.metric('Loss', value=str(loss))

    if multiple == True:
        st.subheader('Multiple Team Selection')
        home_teams = st.multiselect('Select home sides: ', standings['Team'])
        away_teams = st.multiselect('Select away teams: ', standings['Team'])
        batch = st.slider('Select Batch Size', min_value=2, max_value=9)
        multiple_calculate = st.button('Calculate', key='multiple_calculate')
        if multiple_calculate and len(home_teams)==len(away_teams) and len(home_teams)>=1:
            
            win, draw, loss, selected_home, selected_away = accumulator(home_teams, away_teams, fixtures, batch)
            w_col,d_col, l_col = st.columns(3)
            with w_col:
                st.metric('Win', value=win)
            with d_col:
                st.metric('Draw', value=draw)
            with l_col:
                st.metric('Loss', value=loss)
            
            st.dataframe(data=pd.DataFrame.from_dict({
                'Home':selected_home, 'Away':selected_away
            }))
            

    sidebar.subheader('Backtesting')
    run_back_test = sidebar.checkbox('Run Back Test', key='backtest')

    if run_back_test == True:
        # home_side = sidebar.selectbox('Home Team', options=standings['Team'].tolist())
        batch_size = sidebar.slider(label='Selet batch size', min_value=1, max_value=9)
        max_draw = sidebar.slider(label='Maximum draw probability', min_value=0.0, max_value=0.3, step=0.01)
        min_win = sidebar.slider(label='Minimum win probability', min_value=0.2, max_value=1.0, step=0.01)
        run = sidebar.button('Run', key='run')
        if run:
           results = backtest(fixtures, batch_size, max_draw, min_win)
           st.subheader('Simulation Results')
           st.dataframe(results)
    
    optimise_btn = sidebar.button('Derive Optimum Win-Draw Ratios')
    if optimise_btn:
        ops = op.Optimiser(fixtures, 3, 5)
        ops.get_permutations()

if __name__ == "__main__":
    render_UI()


        
