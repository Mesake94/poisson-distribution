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

    return (matrix, home_team_home_games, away_games, home_attack_strength, home_defence_strength, away_attack_strength, away_defence_strength, home_team_goal_expectancy, away_team_goal_expectancy)

@st.cache(suppress_st_warning=True)
def backtest(df, home): # function to back test results and evaluate performance of the poisson distribution
    home_games = df.loc[(df['Home'] == home)] # get all home games
    total_home_games = int(len(home_games))
   
    index = []
    index += [x for x in home_games.index]
    reverse = []
    reverse += [y for y in reversed(home_games.index)]
    
    for itr, _ in enumerate(reverse):
        if itr >= 3:
            print(home_games.loc[reverse[0]:_,:])
            
    # matrix = calculate_odds(home, away_side, fixture_df=prev_)
        
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
        matrix, home_team_home_games, away_games, home_attack_strength, home_defence_strength, away_attack_strength, away_defence_strength, home_team_goal_expectancy, away_team_goal_expectancy = calculate_odds(home, away, fixtures)
        if len(matrix)> 0:
            st.success('Probabilities of possible outcomes successfully calculated.')
            st.subheader('Home Team Games')
            st.dataframe(home_team_home_games ,700, 600)
            st.subheader('Away Team games')
            st.dataframe(away_games, 700, 600)

            st.subheader('General Statistics')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric('Home Attack Strength', value=str(home_attack_strength))
            with col2:
                st.metric('Home Defensive Strength', value=str(home_defence_strength))
            with col3:
                st.metric('Away Attack Strength', value=str(away_attack_strength))
            with col4:
                st.metric('Away Defensive Strength', value=str(away_defence_strength))

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

    sidebar.subheader('Backtesting')
    run_back_test = sidebar.checkbox('Run Back Test', key='backtest')

    if run_back_test == True:
        home_side = sidebar.selectbox('Home Team', options=standings['Team'])
        run = sidebar.button('Run', key='run')
        if run:
            backtest(fixtures, home_side)

if __name__ == "__main__":
    render_UI()
    

        
