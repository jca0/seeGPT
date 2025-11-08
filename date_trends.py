import json 
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
from collections import Counter

with open('conversations.json', 'r') as f:
    conversations = json.load(f)

def get_user_msg_dates(conversations):
    dates = []
    for convo in conversations:
        for node in convo['mapping'].keys():
            if convo['mapping'][node]['message'] and convo['mapping'][node]['message']['author']['role'] == 'user':
                dt = datetime.fromtimestamp(convo['create_time'])
                dates.append(dt)
    return dates

daily_counts = Counter([d.date() for d in get_user_msg_dates(conversations)])
weekly_counts = Counter([d.strftime('%Y-%U') for d in get_user_msg_dates(conversations)])
monthly_counts = Counter([d.strftime('%Y-%m') for d in get_user_msg_dates(conversations)])

daily_sorted_dates = sorted(daily_counts.keys())
weekly_sorted_dates = sorted(weekly_counts.keys())
monthly_sorted_dates = sorted(monthly_counts.keys())

daily_counts = [daily_counts[d] for d in daily_sorted_dates]
weekly_counts = [weekly_counts[d] for d in weekly_sorted_dates]
monthly_counts = [monthly_counts[d] for d in monthly_sorted_dates]


# PLOTTING
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
# Convert monthly date strings to datetime objects for proper formatting
monthly_dates = [datetime.strptime(d, '%Y-%m') for d in monthly_sorted_dates]
plt.plot(monthly_dates, monthly_counts)
plt.title('Number of Messages per Month')
plt.xlabel('Date')
plt.ylabel('Number of Messages')
# Set x-axis to show all months
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()

plt.subplot(3, 1, 2)
# Convert weekly date strings to datetime objects (using first day of each week)
weekly_dates = []
for week_str in weekly_sorted_dates:
    year, week = week_str.split('-')
    year, week = int(year), int(week)
    # Calculate first day of week (%U uses Sunday as first day, week 0 starts on first Sunday)
    jan1 = datetime(year, 1, 1)
    # Find first Sunday
    days_until_sunday = (6 - jan1.weekday()) % 7
    first_sunday = jan1 + timedelta(days=days_until_sunday)
    # Calculate week start
    week_start = first_sunday + timedelta(weeks=week)
    weekly_dates.append(week_start)
plt.plot(weekly_dates, weekly_counts)
plt.title('Number of Messages per Week')
plt.xlabel('Date')
plt.ylabel('Number of Messages')
# Set x-axis to show all months
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()

plt.subplot(3, 1, 3)
# Convert daily date objects to datetime objects for proper formatting
daily_dates = [datetime(d.year, d.month, d.day) for d in daily_sorted_dates]
plt.plot(daily_dates, daily_counts)
plt.title('Number of Messages per Day')
plt.xlabel('Date')
plt.ylabel('Number of Messages')
# Set x-axis to show all months
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()