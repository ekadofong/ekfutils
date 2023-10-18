import os
import datetime
import numpy as np
import pandas as pd

item_type_values = { 
                    'checklist':1., 
                    'procedural':0.1,
                    'exploratory':0.5
                    }

class Tracker (object):
    def __init__(self, logfile=None, ) -> None:
        if logfile is None:
            logfile = os.environ['HOME'] + '/.ekftrack_log'
        self.logfile = logfile
        self.__load_db__ ()
          
    def __load_db__ (self):
        today = pd.to_datetime ( datetime.datetime.today().date() )
        self.today = today
        
        if not os.path.exists ( self.logfile ):            
            multiindex = pd.MultiIndex.from_product ( [[today], list(item_type_values.keys())] )
            df = pd.DataFrame ( np.zeros(len(list(item_type_values.keys()))), index=multiindex, columns=['points'])
        else:
            df = pd.read_csv ( self.logfile, index_col=[0,1] )  
            if today not in df.index:
                for tval in item_type_values.keys():
                    df.loc[(today, tval),:] = 0.
                    
        self.__db__ = df            
    
    def add_item (self):
        pass
    
    def remove_item (self):
        pass

    def check_completion (self):
        pass

    def check_all_items (self):
        pass

