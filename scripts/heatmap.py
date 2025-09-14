"""
TRADING ANALYTICS HEATMAP GENERATOR
===================================

This script performs comprehensive analysis of options trading data from CSV files and generates
multiple visualizations and reports to identify patterns in trading performance.

FEATURES:
---------
1. **Data Loading & Processing**:
   - Automatically finds and loads all CSV files matching '*mei*.csv' pattern in the script directory
   - Combines multiple CSV files into a single dataset for analysis
   - Extracts time intervals, day of week, and put/call information from trade data

2. **Generated Visualizations**:
   - **Heatmap**: Average P&L by time interval and day of week
   - **Day Summary**: Average P&L by day of week with trade counts
   - **Trade Count Analysis**: Number of trades by time interval
   - **Total P&L Analysis**: Sum of P&L by time interval across all days
   - **Stop Loss Heatmaps**: Stop loss frequency by time and day
   - **Put/Call Analysis**: Stop loss distribution between puts and calls
   - **Reason for Close**: Distribution of all trade exit reasons

3. **Data Reports**:
   - Stop loss frequency analysis by date/time combinations
   - Reason for close summary statistics
   - Comprehensive CSV exports of all analysis results

4. **Output Management**:
   - Creates timestamped output folders in 'logs/' directory
   - Copies original CSV files and script to output folder for reproducibility
   - Saves all charts as high-resolution PNG files
   - Exports summary data as CSV files

USAGE:
------
1. Place CSV files with trading data in the same directory as this script
2. Ensure CSV files contain columns: 'Date Opened', 'Time Opened', 'P/L', 'Reason For Close', 'Legs'
3. Run the script: python heatmap.py
4. Results will be saved in: scripts/logs/[filename]_[timestamp]/

REQUIREMENTS:
-------------
- pandas: Data manipulation and analysis
- matplotlib: Chart generation
- seaborn: Statistical visualizations and heatmaps
- numpy: Numerical computations

AUTHOR: OO Analytics Team
DATE: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find all CSV files matching *meic*.csv pattern
import glob
meic_files = glob.glob(os.path.join(script_dir, '*mei*.csv'))
meic_files = sorted(meic_files)  # Sort for consistent ordering

if not meic_files:
    print("No files matching *meic*.csv found in the script directory!")
    exit()

print(f"Found {len(meic_files)} file(s) matching *meic*.csv pattern:")
for file in meic_files:
    print(f"  - {os.path.basename(file)}")

# Create output subfolder based on first CSV filename
first_csv_name = os.path.splitext(os.path.basename(meic_files[0]))[0]  # Remove .csv extension
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_folder = os.path.join(script_dir, "logs", f"{first_csv_name}_{timestamp}")

# Create the output directory
os.makedirs(output_folder, exist_ok=True)
print(f"\nCreated output folder: {output_folder}")

# Copy all CSV files to the output folder
print("\nCopying CSV files to output folder...")
import shutil
for csv_file in meic_files:
    csv_filename = os.path.basename(csv_file)
    dest_path = os.path.join(output_folder, csv_filename)
    shutil.copy2(csv_file, dest_path)
    print(f"  Copied {csv_filename} to output folder")

# Copy the script itself to the output folder
script_filename = os.path.basename(__file__)
script_dest_path = os.path.join(output_folder, script_filename)
shutil.copy2(__file__, script_dest_path)
print(f"  Copied {script_filename} to output folder")

# Load and combine all matching CSV files
dfs = []
for csv_file in meic_files:
    print(f"\nLoading {os.path.basename(csv_file)}...")
    df_temp = pd.read_csv(csv_file)
    print(f"  Loaded {len(df_temp)} rows")
    dfs.append(df_temp)

# Combine all dataframes
df = pd.concat(dfs, ignore_index=True)
print(f"\nCombined dataset: {len(df)} total rows")

print(f"Columns: {list(df.columns)}")

# Display first few rows to understand the data structure
print("\nFirst 5 rows:")
print(df.head())

# Display basic info about the dataset
print("\nDataset info:")
print(df.info())

# Display summary statistics
print("\nSummary statistics:")
print(df.describe())

# Create heatmap of P/L by Time Opened
print("\nCreating heatmap of P/L by Time Opened...")

# Convert Time Opened to datetime and add day of week
df['Time Opened'] = pd.to_datetime(df['Time Opened'], format='%H:%M:%S')
df['Date Opened'] = pd.to_datetime(df['Date Opened'])
df['Day_of_Week'] = df['Date Opened'].dt.day_name()

# Dynamically determine time intervals from the data
time_intervals = df['Time Opened'].dt.strftime('%H:%M').unique()
time_intervals = sorted(time_intervals)
print(f"Found {len(time_intervals)} unique time intervals in the data")
print(f"Time range: {time_intervals[0]} to {time_intervals[-1]}")

# Use the exact time intervals from the data
df['Time_Interval'] = df['Time Opened'].dt.strftime('%H:%M')

# Extract put/call information from Legs column
df['Is_Put'] = df['Legs'].str.contains(' P ', case=False)
df['Is_Call'] = df['Legs'].str.contains(' C ', case=False)

# Show day of week distribution
print(f"\nDay of week distribution:")
print(df['Day_of_Week'].value_counts().sort_index())

# Show stop loss analysis
print(f"\nStop Loss Analysis:")
stop_loss_trades = df[df['Reason For Close'] == 'Stop Loss']
print(f"Total stop loss trades: {len(stop_loss_trades)}")
print(f"Stop loss percentage: {len(stop_loss_trades)/len(df)*100:.1f}%")

if len(stop_loss_trades) > 0:
    put_stop_loss = stop_loss_trades[stop_loss_trades['Is_Put'] == True]
    call_stop_loss = stop_loss_trades[stop_loss_trades['Is_Call'] == True]
    print(f"Put side stop losses: {len(put_stop_loss)} ({len(put_stop_loss)/len(stop_loss_trades)*100:.1f}%)")
    print(f"Call side stop losses: {len(call_stop_loss)} ({len(call_stop_loss)/len(stop_loss_trades)*100:.1f}%)")

# Create pivot table for time intervals and day of week
print("Creating heatmap by time intervals and day of week...")

pivot_data = df.pivot_table(
    values='P/L', 
    index='Time_Interval', 
    columns='Day_of_Week', 
    aggfunc='mean',
    fill_value=0
)

# Reorder columns to show days in chronological order
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot_data = pivot_data.reindex(columns=[day for day in day_order if day in pivot_data.columns])

# Create heatmap visualization
plt.figure(figsize=(16, 10))
sns.heatmap(pivot_data, 
            annot=True, 
            fmt='.0f', 
            cmap='RdYlGn', 
            center=0,
            cbar_kws={'label': 'Average P/L'})

plt.title('Average P/L by Time Interval and Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Time of Day')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()

# Save heatmap to PNG file
heatmap_filename = os.path.join(output_folder, 'heatmap_time_day_pl.png')
plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
print(f"Saved heatmap to: {heatmap_filename}")

plt.show(block=False)
plt.pause(0.1)  # Small pause to ensure plot renders

# Also create a summary by day of week
print("\nCreating day of week summary...")
day_summary = df.groupby('Day_of_Week')['P/L'].agg(['mean', 'count']).reset_index()
day_summary.columns = ['Day_of_Week', 'Avg_PL', 'Trade_Count']

# Reorder by day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_summary['Day_of_Week'] = pd.Categorical(day_summary['Day_of_Week'], categories=day_order, ordered=True)
day_summary = day_summary.sort_values('Day_of_Week')

plt.figure(figsize=(12, 6))
colors = ['green' if x >= 0 else 'red' for x in day_summary['Avg_PL']]
bars = plt.bar(range(len(day_summary)), day_summary['Avg_PL'], color=colors, alpha=0.7)

# Add trade count as text on bars
for i, (bar, count) in enumerate(zip(bars, day_summary['Trade_Count'])):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + (0.01 * max(day_summary['Avg_PL'])),
             f'n={count}', ha='center', va='bottom', fontsize=10)

plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title('Average P/L by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average P/L')
plt.xticks(range(len(day_summary)), day_summary['Day_of_Week'], rotation=45, ha='right')
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save day summary chart to PNG file
day_summary_filename = os.path.join(output_folder, 'day_summary_pl.png')
plt.savefig(day_summary_filename, dpi=300, bbox_inches='tight')
print(f"Saved day summary chart to: {day_summary_filename}")

plt.show(block=False)
plt.pause(0.1)  # Small pause to ensure plot renders



# Create additional statistics and visualizations
print("\nCreating additional statistics and visualizations...")

# 1. Trade count per time slice
print("\n1. Creating trade count per time slice chart...")
time_counts = df.groupby('Time_Interval').size().reset_index(name='Trade_Count')
time_counts = time_counts.sort_values('Time_Interval')

plt.figure(figsize=(15, 6))
bars = plt.bar(range(len(time_counts)), time_counts['Trade_Count'], alpha=0.7, color='skyblue')
plt.title('Trade Count by Time Interval')
plt.xlabel('Time of Day')
plt.ylabel('Number of Trades')
plt.xticks(range(len(time_counts)), time_counts['Time_Interval'], rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Add trade count as text on bars
for i, (bar, count) in enumerate(zip(bars, time_counts['Trade_Count'])):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + max(time_counts['Trade_Count'])*0.01,
             f'{count}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
trade_count_filename = os.path.join(output_folder, 'trade_count_by_time.png')
plt.savefig(trade_count_filename, dpi=300, bbox_inches='tight')
print(f"Saved trade count chart to: {trade_count_filename}")
plt.show(block=False)
plt.pause(0.1)  # Small pause to ensure plot renders

# 1b. Total P&L per time slice (sum across all days)
print("\n1b. Creating total P&L per time slice chart...")
time_pl_sum = df.groupby('Time_Interval')['P/L'].sum().reset_index()
time_pl_sum = time_pl_sum.sort_values('Time_Interval')

plt.figure(figsize=(15, 6))
colors = ['green' if x >= 0 else 'red' for x in time_pl_sum['P/L']]
bars = plt.bar(range(len(time_pl_sum)), time_pl_sum['P/L'], color=colors, alpha=0.7)
plt.title('Total P&L by Time Interval (Sum Across All Days)')
plt.xlabel('Time of Day')
plt.ylabel('Total P&L ($)')
plt.xticks(range(len(time_pl_sum)), time_pl_sum['Time_Interval'], rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Add P&L values as text on bars
for i, (bar, pl) in enumerate(zip(bars, time_pl_sum['P/L'])):
    height = bar.get_height()
    # Position text above or below bar based on P&L value
    if pl >= 0:
        plt.text(bar.get_x() + bar.get_width()/2., height + max(abs(time_pl_sum['P/L']))*0.01,
                 f'${pl:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        plt.text(bar.get_x() + bar.get_width()/2., height - max(abs(time_pl_sum['P/L']))*0.01,
                 f'${pl:,.0f}', ha='center', va='top', fontsize=8, fontweight='bold')

plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
time_pl_sum_filename = os.path.join(output_folder, 'total_pl_by_time.png')
plt.savefig(time_pl_sum_filename, dpi=300, bbox_inches='tight')
print(f"Saved total P&L by time chart to: {time_pl_sum_filename}")
plt.show(block=False)
plt.pause(0.1)  # Small pause to ensure plot renders

# 1c. Summary table of P&L by time slice (both average and sum)
print("\n1c. Creating P&L summary table by time slice...")
time_pl_summary = df.groupby('Time_Interval')['P/L'].agg(['sum', 'mean', 'count']).reset_index()
time_pl_summary.columns = ['Time_Interval', 'Total_PL', 'Avg_PL', 'Trade_Count']
time_pl_summary = time_pl_summary.sort_values('Time_Interval')

print("\nP&L Summary by Time Interval:")
print("=" * 80)
print(f"{'Time':<15} {'Total P&L':<15} {'Avg P&L':<15} {'Trades':<10}")
print("-" * 80)
for _, row in time_pl_summary.iterrows():
    total_pl = f"${row['Total_PL']:,.0f}"
    avg_pl = f"${row['Avg_PL']:,.0f}"
    trades = f"{row['Trade_Count']}"
    print(f"{row['Time_Interval']:<15} {total_pl:<15} {avg_pl:<15} {trades:<10}")
print("=" * 80)

# 2. Stop Loss Analysis by Time and Day of Week
print("\n2. Creating stop loss analysis by time and day of week...")

# Create pivot table for stop losses by time and day
stop_loss_pivot = df[df['Reason For Close'] == 'Stop Loss'].pivot_table(
    values='P/L', 
    index='Time_Interval', 
    columns='Day_of_Week', 
    aggfunc='count',
    fill_value=0
)

# Reorder columns to show days in chronological order
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
stop_loss_pivot = stop_loss_pivot.reindex(columns=[day for day in day_order if day in stop_loss_pivot.columns])

if not stop_loss_pivot.empty and stop_loss_pivot.sum().sum() > 0:
    # Create heatmap for stop losses by time and day
    plt.figure(figsize=(16, 10))
    sns.heatmap(stop_loss_pivot, 
                annot=True, 
                fmt='.0f', 
                cmap='Reds', 
                cbar_kws={'label': 'Stop Loss Count'})

    plt.title('Stop Loss Count by Time Interval and Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Time of Day')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    stop_loss_heatmap_filename = os.path.join(output_folder, 'stop_loss_heatmap_time_day.png')
    plt.savefig(stop_loss_heatmap_filename, dpi=300, bbox_inches='tight')
    print(f"Saved stop loss heatmap to: {stop_loss_heatmap_filename}")
    plt.show(block=False)
    plt.pause(0.1)  # Small pause to ensure plot renders
    
    # Also create simple time-based chart
    stop_loss_by_time = df[df['Reason For Close'] == 'Stop Loss'].groupby('Time_Interval').size().reset_index(name='Stop_Loss_Count')
    stop_loss_by_time = stop_loss_by_time.sort_values('Time_Interval')
    
    plt.figure(figsize=(15, 6))
    bars = plt.bar(range(len(stop_loss_by_time)), stop_loss_by_time['Stop_Loss_Count'], alpha=0.7, color='red')
    plt.title('Total Stop Loss Count by Time Interval')
    plt.xlabel('Time of Day')
    plt.ylabel('Number of Stop Losses')
    plt.xticks(range(len(stop_loss_by_time)), stop_loss_by_time['Time_Interval'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add count as text on bars
    for i, (bar, count) in enumerate(zip(bars, stop_loss_by_time['Stop_Loss_Count'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(stop_loss_by_time['Stop_Loss_Count'])*0.01,
                 f'{count}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    stop_loss_filename = os.path.join(output_folder, 'stop_loss_by_time.png')
    plt.savefig(stop_loss_filename, dpi=300, bbox_inches='tight')
    print(f"Saved stop loss time chart to: {stop_loss_filename}")
    plt.show(block=False)
    plt.pause(0.1)  # Small pause to ensure plot renders
    
    # Create day of week stop loss summary
    print("\n2a. Creating stop loss by day of week...")
    stop_loss_by_day = df[df['Reason For Close'] == 'Stop Loss'].groupby('Day_of_Week').size().reset_index(name='Stop_Loss_Count')
    stop_loss_by_day['Day_of_Week'] = pd.Categorical(stop_loss_by_day['Day_of_Week'], categories=day_order, ordered=True)
    stop_loss_by_day = stop_loss_by_day.sort_values('Day_of_Week')
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(stop_loss_by_day)), stop_loss_by_day['Stop_Loss_Count'], alpha=0.7, color='darkred')
    plt.title('Stop Loss Count by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Stop Losses')
    plt.xticks(range(len(stop_loss_by_day)), stop_loss_by_day['Day_of_Week'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add count as text on bars
    for i, (bar, count) in enumerate(zip(bars, stop_loss_by_day['Stop_Loss_Count'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(stop_loss_by_day['Stop_Loss_Count'])*0.01,
                 f'{count}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    stop_loss_day_filename = os.path.join(output_folder, 'stop_loss_by_day.png')
    plt.savefig(stop_loss_day_filename, dpi=300, bbox_inches='tight')
    print(f"Saved stop loss by day chart to: {stop_loss_day_filename}")
    plt.show(block=False)
    plt.pause(0.1)  # Small pause to ensure plot renders
    
else:
    print("No stop loss trades found in the data.")

# 3. Put vs Call Stop Loss Analysis
print("\n3. Creating put vs call stop loss analysis...")
if len(stop_loss_trades) > 0:
    put_call_stop_loss = pd.DataFrame({
        'Side': ['Put', 'Call'],
        'Count': [len(put_stop_loss), len(call_stop_loss)]
    })
    
    plt.figure(figsize=(10, 6))
    colors = ['orange', 'blue']
    bars = plt.bar(put_call_stop_loss['Side'], put_call_stop_loss['Count'], color=colors, alpha=0.7)
    plt.title('Stop Losses by Option Side')
    plt.xlabel('Option Side')
    plt.ylabel('Number of Stop Losses')
    plt.grid(True, alpha=0.3)
    
    # Add count as text on bars
    for i, (bar, count) in enumerate(zip(bars, put_call_stop_loss['Count'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(put_call_stop_loss['Count'])*0.01,
                 f'{count}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    put_call_filename = os.path.join(output_folder, 'put_call_stop_loss.png')
    plt.savefig(put_call_filename, dpi=300, bbox_inches='tight')
    print(f"Saved put/call stop loss chart to: {put_call_filename}")
    plt.show(block=False)
    plt.pause(0.1)  # Small pause to ensure plot renders

# 4. Reason for Close Distribution
print("\n4. Creating reason for close distribution...")
reason_counts = df['Reason For Close'].value_counts()
total_trades = len(df)

# Print summary statistics
print("\nReason for Close Summary Statistics:")
print("=" * 60)
print(f"{'Reason':<20} {'Count':<8} {'Percentage':<12}")
print("-" * 60)
for reason, count in reason_counts.items():
    percentage = (count / total_trades) * 100
    print(f"{reason:<20} {count:<8} {percentage:<12.1f}%")
print("-" * 60)
print(f"{'TOTAL':<20} {total_trades:<8} {'100.0%':<12}")
print("=" * 60)

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(reason_counts)), reason_counts.values, alpha=0.7, color='lightgreen')
plt.title('Distribution of Reasons for Close')
plt.xlabel('Reason for Close')
plt.ylabel('Number of Trades')
plt.xticks(range(len(reason_counts)), reason_counts.index, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Add count and percentage as text on bars
for i, (bar, count) in enumerate(zip(bars, reason_counts.values)):
    height = bar.get_height()
    percentage = (count / total_trades) * 100
    plt.text(bar.get_x() + bar.get_width()/2., height + max(reason_counts.values)*0.01,
             f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
reason_filename = os.path.join(output_folder, 'reason_for_close_distribution.png')
plt.savefig(reason_filename, dpi=300, bbox_inches='tight')
print(f"Saved reason for close chart to: {reason_filename}")
plt.show(block=False)
plt.pause(0.1)  # Small pause to ensure plot renders

# Save reason for close summary to CSV
reason_summary_data = []
for reason, count in reason_counts.items():
    percentage = (count / total_trades) * 100
    reason_summary_data.append({
        'Reason_For_Close': reason,
        'Count': count,
        'Percentage': round(percentage, 1)
    })

reason_summary_df = pd.DataFrame(reason_summary_data)
reason_summary_csv = os.path.join(output_folder, 'reason_for_close_summary.csv')
reason_summary_df.to_csv(reason_summary_csv, index=False)
print(f"Saved reason for close summary to: {reason_summary_csv}")

# 5. Stop Loss Frequency Report by Date/Time
print("\n5. Creating stop loss frequency report by date/time...")

# Group by Date and Time to create unique date/time combinations
df['DateTime'] = df['Date Opened'].astype(str) + ' ' + df['Time Opened'].astype(str)
datetime_groups = df.groupby(['Date Opened', 'Time Opened'])

stop_loss_frequency_data = []

for (date, time), group in datetime_groups:
    # Separate puts and calls in this date/time group
    puts_in_group = group[group['Is_Put'] == True]
    calls_in_group = group[group['Is_Call'] == True]
    
    # Check if any puts or calls were stopped out
    puts_stopped = len(puts_in_group[puts_in_group['Reason For Close'] == 'Stop Loss']) > 0
    calls_stopped = len(calls_in_group[calls_in_group['Reason For Close'] == 'Stop Loss']) > 0
    
    # Determine status
    if puts_stopped and calls_stopped:
        status = 'Both'
    elif puts_stopped:
        status = 'Puts Only'
    elif calls_stopped:
        status = 'Calls Only'
    else:
        status = 'None'
    
    # Add separate entries for puts and calls
    # Format time to show only HH:MM
    time_formatted = pd.to_datetime(time).strftime('%H:%M') if pd.notnull(time) else str(time)
    
    if len(puts_in_group) > 0:
        stop_loss_frequency_data.append({
            'Date': date,
            'Time': time_formatted,
            'Side': 'Put',
            'Stopped_Out': puts_stopped,
            'Status': status
        })
    
    if len(calls_in_group) > 0:
        stop_loss_frequency_data.append({
            'Date': date,
            'Time': time_formatted,
            'Side': 'Call', 
            'Stopped_Out': calls_stopped,
            'Status': status
        })

# Convert to DataFrame
stop_loss_freq_df = pd.DataFrame(stop_loss_frequency_data)

if len(stop_loss_freq_df) > 0:
    # Calculate frequency counts and percentages
    print("\nStop Loss Frequency Report:")
    print("=" * 80)
    print("(Each date/time treated as single 'trade' with separate put/call entries)")
    print("-" * 80)
    
    # Overall statistics by status
    status_counts = stop_loss_freq_df['Status'].value_counts()
    total_entries = len(stop_loss_freq_df)
    
    print(f"\nOverall Status Distribution (Total Entries: {total_entries}):")
    print("-" * 40)
    for status in ['None', 'Puts Only', 'Calls Only', 'Both']:
        count = status_counts.get(status, 0)
        percentage = (count / total_entries * 100) if total_entries > 0 else 0
        print(f"{status:<12}: {count:>4} ({percentage:>5.1f}%)")
    
    # Statistics by side
    print(f"\nBy Option Side:")
    print("-" * 40)
    put_entries = stop_loss_freq_df[stop_loss_freq_df['Side'] == 'Put']
    call_entries = stop_loss_freq_df[stop_loss_freq_df['Side'] == 'Call']
    
    put_stopped_count = len(put_entries[put_entries['Stopped_Out'] == True])
    call_stopped_count = len(call_entries[call_entries['Stopped_Out'] == True])
    
    put_percentage = (put_stopped_count / len(put_entries) * 100) if len(put_entries) > 0 else 0
    call_percentage = (call_stopped_count / len(call_entries) * 100) if len(call_entries) > 0 else 0
    
    print(f"Put Stops   : {put_stopped_count:>4} / {len(put_entries):>4} ({put_percentage:>5.1f}%)")
    print(f"Call Stops  : {call_stopped_count:>4} / {len(call_entries):>4} ({call_percentage:>5.1f}%)")
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Status distribution
    plt.subplot(2, 2, 1)
    status_order = ['None', 'Puts Only', 'Calls Only', 'Both']
    status_counts_ordered = [status_counts.get(status, 0) for status in status_order]
    colors = ['lightgreen', 'orange', 'lightblue', 'red']
    
    bars = plt.bar(status_order, status_counts_ordered, color=colors, alpha=0.7)
    plt.title('Stop Loss Status Distribution\n(by Date/Time Entry)')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, status_counts_ordered):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(status_counts_ordered)*0.01,
                 f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 2: Put vs Call stop rates
    plt.subplot(2, 2, 2)
    side_data = ['Puts', 'Calls']
    side_stopped = [put_stopped_count, call_stopped_count]
    side_total = [len(put_entries), len(call_entries)]
    side_percentages = [put_percentage, call_percentage]
    
    bars = plt.bar(side_data, side_percentages, color=['orange', 'lightblue'], alpha=0.7)
    plt.title('Stop Loss Rate by Option Side')
    plt.ylabel('Stop Loss Percentage (%)')
    
    # Add percentage labels on bars
    for bar, pct, stopped, total in zip(bars, side_percentages, side_stopped, side_total):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(side_percentages)*0.01,
                 f'{pct:.1f}%\n({stopped}/{total})', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 3: Time-based heatmap of stop loss status
    plt.subplot(2, 1, 2)
    
    # Create pivot for heatmap showing status by time
    pivot_status = stop_loss_freq_df.pivot_table(
        values='Stopped_Out', 
        index='Time', 
        columns='Side', 
        aggfunc='sum',
        fill_value=0
    )
    
    if not pivot_status.empty:
        sns.heatmap(pivot_status, 
                    annot=True, 
                    fmt='.0f', 
                    cmap='Reds',
                    cbar_kws={'label': 'Stop Loss Count'})
        plt.title('Stop Loss Count by Time and Option Side')
        plt.xlabel('Option Side')
        plt.ylabel('Time of Day')
    
    plt.tight_layout()
    
    # Save the stop loss frequency report chart
    stop_loss_freq_filename = os.path.join(output_folder, 'stop_loss_frequency_report.png')
    plt.savefig(stop_loss_freq_filename, dpi=300, bbox_inches='tight')
    print(f"Saved stop loss frequency report to: {stop_loss_freq_filename}")
    plt.show(block=False)
    plt.pause(0.1)  # Small pause to ensure plot renders
    
    # Save detailed data to CSV
    stop_loss_freq_csv = os.path.join(output_folder, 'stop_loss_frequency_data.csv')
    stop_loss_freq_df.to_csv(stop_loss_freq_csv, index=False)
    print(f"Saved stop loss frequency data to: {stop_loss_freq_csv}")
    
    print("=" * 80)

print(f"\nAnalysis complete!")
print(f"Total trades: {len(df)}")
print(f"Date range: {df['Date Opened'].min()} to {df['Date Opened'].max()}")
print(f"Days of week found: {df['Day_of_Week'].nunique()}")
print(f"Unique days: {sorted(df['Day_of_Week'].unique())}")
print(f"Total stop losses: {len(stop_loss_trades)} ({len(stop_loss_trades)/len(df)*100:.1f}%)")

print(f"\n" + "="*80)
print(f"ALL FILES SAVED TO: {output_folder}")
print(f"="*80)
print(f"Files copied to output folder:")
for csv_file in meic_files:
    print(f"  - {os.path.basename(csv_file)}")
print(f"  - {os.path.basename(__file__)}")
print(f"\nGenerated charts:")
print(f"  - heatmap_time_day_pl.png")
print(f"  - day_summary_pl.png")
print(f"  - trade_count_by_time.png")
print(f"  - total_pl_by_time.png")
print(f"  - stop_loss_heatmap_time_day.png")
print(f"  - stop_loss_by_time.png")
print(f"  - stop_loss_by_day.png")
print(f"  - put_call_stop_loss.png")
print(f"  - reason_for_close_distribution.png")
print(f"  - stop_loss_frequency_report.png")
print(f"\nGenerated data files:")
print(f"  - stop_loss_frequency_data.csv")
print(f"  - reason_for_close_summary.csv")
print(f"\nOutput folder: {output_folder}")
print(f"="*80)