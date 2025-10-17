# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 14:49:56 2025

@author: bswan work
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the TSA checkpoint travel numbers page
url = 'https://www.tsa.gov/travel/passenger-volumes'

response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    

    table = soup.find('table')
    
    if table:
        # Create lists to store the data!
        dates = []
        numbers = []
        
        # Skip the header row by starting with tbody rows
        rows = table.find('tbody').find_all('tr')
        
        for row in rows:
            # Extract the date and number from each row
            cells = row.find_all('td')
            if len(cells) >= 2:
                date = cells[0].text.strip()
                number = cells[1].text.strip()
                dates.append(date)
                numbers.append(number)
        
        # Create a DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Numbers': numbers
        })
        
        # Convert Numbers to numeric, removing commas
        df['Numbers'] = df['Numbers'].str.replace(',', '').astype(int)
        
        # Save to CSV
        df.to_csv('tsa_checkpoint_data.csv', index=False)
        print("Data successfully scraped and saved to tsa_checkpoint_data.csv")
    else:
        print("Table not found on the page.")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")