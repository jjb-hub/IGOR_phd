
import sqlite3
import pandas as pd

from module.getters import getRawDf


#will iniate or replace 'folder_file' table in db
def initiate_ephys_phd_db(filename):
    #get raw df
    df = getRawDf(filename)

    #connection to SQLite database
    conn = sqlite3.connect('ephys_phd.db')

    #convert expanded df to sql database table 'file_follder' as it is a unique identrifier for each row
    df.to_sql('folder_file', conn, if_exists='replace', index=False)

    # Commit changes and close the connection
    conn.commit()
    conn.close()

    initiate_ephys_phd_db(filename)

#### humm this would be fucking complex probaby have a feature that goes in 
# then a table for each thing extracted from the file ie AP_height if data_type='FP' 
# then a table for each cell_type and drug that corrisponds to a treatment group
#  .... where each line is then a cell_id (linked to folder_file via feature table) 
# then i would extract and minipulate that data to do stats and make a metta table with stats between treatments

# check out : ORM (Object-Relational Mapping)
#https://www.tutorialspoint.com/sqlite/sqlite_python.htm


#define the class for a single treatment group
class TreatmentGroup:
    def __init__(self, cell_type, drug):
        self.cell_type = cell_type
        self.drug = drug

    def fetch_data_from_db(self):
        conn = sqlite3.connect('ephys_phd.db')  # Connect to your database
        cursor = conn.cursor()
        
        # Example query: Fetch data for a specific cell_type and drug combination
        query = f"SELECT * FROM folder_file WHERE cell_type = ? AND drug = ?"
        cursor.execute(query, (self.cell_type, self.drug))
        
        # Fetch the data
        data = cursor.fetchall()
        
        # Close the connection
        conn.close()
        
        return data


whatisthis= TreatmentGroup('L5a_TLX', 'LSD').fetch_data_from_db()

