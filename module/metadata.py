####### FUNCTION THAT DEAL WITH SAVING INFORMATIONS ABOUT EXPERIMENTS ###########
from module.utils import saveDataTracking
import json


insufficient_data_tracking = {}

# Save it to a JSON file
with open('insufficient_data_tracking.json', 'w') as file:
    json.dump(insufficient_data_tracking, file, indent=4)
