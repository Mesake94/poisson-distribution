from typing import Tuple
import numpy as np
import itertools

from numpy.core.fromnumeric import shape

class Optimiser:
    # variables accessible anythere inside the class
    win_array = np.array([1,0,0])
    draw_array = np.array([1,1,0])
    lose_array = np.array([0,0,1])

    def __init__(self, fixtures, batch:int, start_point:int, *args, **kwargs):
        self.fixtures = fixtures # get the fixtures
        self.batch = batch # represents the number of teams in a selection
        self.start_point = start_point # represents the fixture date to start the calculation from; should be greater than atleast 3 to ensure enough games for each team has passed

    def get_permutations(self) -> Tuple:
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
                    
                    for key, value in sorted_perms.items():
                        if key not in result.keys():
                            result[key] = value

                        
                    arr_perms.append(result)
                    selected_dates.append(dt)
        
        return (arr_perms, selected_dates)


    def calculate_probabilities(self):

        combinations, dates = self.get_permutations()
        

                    
                    