from dotenv import load_dotenv
import os
load_dotenv()
SUMO_HOME = os.getenv('Sumo_Home')
print(SUMO_HOME)
