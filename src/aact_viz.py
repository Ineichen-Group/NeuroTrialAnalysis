import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable

def viz_purpose_status(trial_metadata, output_file):
    # Keeping only unique nct_id, primary_purpose pairs
    unique_pairs_purpose = trial_metadata[['nct_id', 'primary_purpose']].drop_duplicates()

    # Counting the number of nct_ids per primary purpose type
    purpose_type_counts = unique_pairs_purpose['primary_purpose'].value_counts()
    purpose_type_counts = purpose_type_counts.sort_values(ascending=True)

    # Keeping only unique nct_id, overall_status pairs
    unique_pairs_status = trial_metadata[['nct_id', 'overall_status']].drop_duplicates()

    # Counting the number of nct_ids per overall status type
    status_type_counts = unique_pairs_status['overall_status'].value_counts()
    status_type_counts = status_type_counts.sort_values(ascending=True)

    # Create a figure with two horizontal bar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Plot for Primary Purpose
    ax1.barh(purpose_type_counts.index, purpose_type_counts, color='lightgrey', zorder=2)
    for i, v in enumerate(purpose_type_counts):
        ax1.text(v + 50, i, str(v), va='center', color='black', fontsize=12)
    ax1.set_title('Primary Trial Purpose', fontsize=14)
    ax1.set_xlabel('Count of Unique Trials', fontsize=14)
    ax1.set_xlim(0, max(purpose_type_counts)+1500)  # Adjusting the x limits for visibility
    ax1.grid(axis='x', linestyle='--', alpha=0.6, zorder=1)
    ax1.set_xlim(0, max(purpose_type_counts)+1800) # Adjusted to max count for relevancy
    ax1.tick_params(axis='both', labelsize=13)  # Increase tick label size
    ax1.text(-0.03, 1.05, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # Plot for Overall Status
    ax2.barh(status_type_counts.index, status_type_counts, color='lightgrey', zorder=2)
    for i, v in enumerate(status_type_counts):
        ax2.text(v + 50, i, str(v), va='center', color='black', fontsize=12)
    ax2.set_title('Overall Trial Status', fontsize=14)
    ax2.set_xlabel('Count of Unique Trials', fontsize=14)
    ax2.set_xlim(0, max(status_type_counts)+1000)  # Adjusting the x limits for visibility
    ax2.grid(axis='x', linestyle='--', alpha=0.6, zorder=1)
    ax2.set_xlim(0, max(status_type_counts)+1500) # Adjusted to max count for relevancy
    ax2.tick_params(axis='both', labelsize=13)  # Increase tick label size
    ax2.text(-0.03, 1.05, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    # Adjust layout and display the plots
    plt.tight_layout()

    # Optionally save the figure to a local folder
    fig.savefig(output_file)

def viz_phase_year_growth(trial_metadata, output_file):
    # Define phase order and filter/preprocess the data
    phase_order = [
        'Early Phase 1',
        'Phase 1',
        'Phase 1/2',
        'Phase 2',
        'Phase 2/3',
        'Phase 3',
        'Phase 4',
        'Not Applicable'
    ]

    # Define custom colors for each phase
    phase_colors = {
        'Early Phase 1': '#882255',
        'Phase 1': '#AA4499',
        'Phase 1/2': '#CC6677',
        'Phase 2': '#DDCC77',
        'Phase 2/3': '#88CCEE',
        'Phase 3': '#44AA99',
        'Phase 4': '#117733',
        'Not Applicable': '#332288'
    }

    # Filter and count phases and statuses
    unique_pairs_phase = trial_metadata[['nct_id', 'phase', 'overall_status']].drop_duplicates()
    unique_pairs_phase['phase'] = unique_pairs_phase['phase'].str.replace('/Phase ', '/', regex=False)

    phase_type_counts = unique_pairs_phase['phase'].value_counts().reindex(phase_order, fill_value=0)
    completed_count = unique_pairs_phase[unique_pairs_phase['overall_status'] == 'Completed']['phase'].value_counts().reindex(phase_order, fill_value=0)
    completed_proportion = (completed_count / phase_type_counts * 100).fillna(0)

    # Prepare data for the time series plot
    filtered_data = trial_metadata[['nct_id', 'phase', 'start_year']][trial_metadata['start_year'] < 2024].drop_duplicates()
    trial_counts = filtered_data.groupby(['phase', 'start_year']).size().unstack(fill_value=0)
    total_trials_per_year = trial_counts.sum(axis=0)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

    # First subplot - Completion by phase
    total_bars = ax1.bar(phase_type_counts.index, phase_type_counts, color='lightgrey', zorder=2, label='Total Trials')
    completed_bars = ax1.bar(completed_count.index, completed_count, color='darkgrey', zorder=2, label='Completed Trials')

    # Label the bars
    for bar, prop in zip(completed_bars, completed_proportion):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{prop:.1f}%', ha='center', va='bottom', fontsize=10)

    for bar in total_bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=12)

    ax1.set_xlabel('Trial Phase',fontsize=14)
    ax1.set_title('Trial Completion by Phase',fontsize=14)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.text(-0.03, 1.05, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    ax1.tick_params(axis='both', labelsize=12)  # Increase tick label size

    # Second subplot - Trials over time
    ax2.fill_between(total_trials_per_year.index, total_trials_per_year, color='lightgray', alpha=0.3, label='Total Trials')
    for phase in phase_order:
        if phase in trial_counts.index:
            ax2.plot(trial_counts.columns, trial_counts.loc[phase], label=phase, color=phase_colors[phase])

    ax2.set_xlabel('Start Year', fontsize=14)
    ax2.set_title('Count of Trials Started by Phase and Year',fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(linestyle='--', alpha=0.6, zorder=1)
    ax2.set_xticks(np.arange(min(trial_counts.columns), max(trial_counts.columns)+1, 5))
    ax2.text(-0.03, 1.05, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    ax2.tick_params(axis='both', labelsize=13)  # Increase tick label size
    # Adjust layout and display the plots
    plt.tight_layout()

    # Optionally save the figure to a local folder
    fig.savefig(output_file)
    
def viz_countries_world_map(trial_metadata, output_file):
    unique_pairs_facility_countries = trial_metadata[['nct_id', 'country']].drop_duplicates()
    unique_pairs_facility_countries['country'] = unique_pairs_facility_countries['country'].replace({'United States': 'United States of America'})
    unique_pairs_facility_countries['country'] = unique_pairs_facility_countries['country'].replace({'Russian Federation': 'Russia'})
    unique_pairs_facility_countries['country'] = unique_pairs_facility_countries['country'].replace({'Korea, Republic of': 'South Korea'})

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    df_filtered = unique_pairs_facility_countries[unique_pairs_facility_countries['country'] != 'not reported']

    country_counts = df_filtered['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']

    # Load world map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Merge the world GeoDataFrame with the country counts DataFrame
    world = world.merge(country_counts, how="left", left_on="name", right_on="country")

    # Apply a logarithmic transformation to the 'Frequency' column to deal with wide ranges in data
    world['log_count'] = np.log1p(world['count'])

    # Ensure that areas with zero (log1p(0) = 0) are left white by treating them as NaN
    world['log_count'] = world['log_count'].replace(0, np.nan)

    # Plot the data
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    base = world.plot(ax=ax, column='log_count', cmap='Blues', legend=False, legend_kwds={'label': "Number of Clinical Trials by Country", 'orientation': "horizontal"})
    #ctx.add_basemap(ax, crs=world.crs.to_string(), source=ctx.providers.Stamen.Terrain)
    ax.set_axis_off()
    plt.title('World Map with of Clinical Trial Frequency', fontsize=15)

    # Create a custom colorbar
    norm = Normalize(vmin=world['log_count'].min(), vmax=world['log_count'].max())
    sm = ScalarMappable(cmap='Blues', norm=norm)
    sm._A = []  # Fake up the array of the scalar mappable.
    cb = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cb.set_label('Number of Clinical Trials by Country', fontsize=14)

    # Format the ticks to show the actual counts
    tick_locs = np.linspace(world['log_count'].min(), world['log_count'].max(), num=5)
    cb.set_ticks(tick_locs)
    cb.set_ticklabels((np.exp(tick_locs) - 1).round().astype(int))  # Convert log count back to count

    # Save the figure as a PDF
    plt.savefig(output_file)
    
def viz_allocation(trial_design, output_file):
    unique_rows = trial_design[['nct_id', 'allocation']].drop_duplicates()

    # Counting the number of nct_ids per phase type
    allocation_counts = unique_rows['allocation'].value_counts()
    allocation_counts = allocation_counts.sort_values(ascending=True)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot for Allocation (Plot A)
    bars_0 = axs[0].barh(allocation_counts.index, allocation_counts, color='lightgrey', zorder=2)
    for bar in bars_0:
        width = bar.get_width()
        axs[0].text(width, bar.get_y() + bar.get_height() / 2, f'{width}', va='center', fontsize=12)
    axs[0].tick_params(axis='y', labelsize=14)
    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].grid(axis='x', linestyle='--', alpha=0.6, zorder=1)
    axs[0].set_xlabel('Count of Unique Trials', fontsize=14)
    axs[0].set_title('Allocation', fontsize=14)
    axs[0].set_xlim(0, max(allocation_counts) + 1500)  # Adjusted to max count for relevancy
    axs[0].text(-0.02, 1.09, 'A', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    ### PLOT B
    allocation_colors = {'Randomized': '#44AA99',
    'Non-Randomized': '#CC6677',
    'not reported': '#DDCC77'}
    allocation_types = ['Non-Randomized', 'not reported', 'Randomized']

    # Keeping only unique nct_id, allocation pairs
    unique_pairs_allocation = trial_design[['nct_id', 'allocation', 'start_year']].drop_duplicates()

    # Group by start_year and allocation, then count unique nct_ids
    allocation_over_time = unique_pairs_allocation.groupby(['start_year', 'allocation']).size().unstack(fill_value=0)

    # Calculate the proportion of each allocation type from all trials for each year
    allocation_proportion_over_time = allocation_over_time.div(allocation_over_time.sum(axis=1), axis=0)

    # Plot for Allocation Over Time (Plot B)
    bottom = np.zeros(len(allocation_proportion_over_time))
    for allocation_type in allocation_types:
        bars = axs[1].bar(allocation_proportion_over_time.index, 
                        allocation_proportion_over_time[allocation_type], 
                        bottom=bottom, 
                        label=allocation_type,
                        color=allocation_colors.get(allocation_type, 'gray'), zorder=2)  # Use custom color or gray if not specified
        bottom += allocation_proportion_over_time[allocation_type]

        # Add labels to each segment
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axs[1].text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{height:.0%}', ha='center', va='center', fontsize=10, rotation=90)

    axs[1].set_xlabel('Trial Start Year', fontsize=14)
    axs[1].set_title('Proportion of Reported Allocation Over Time', fontsize=14)
    # Set y-axis limits and labels
    axs[1].set_ylim(0, 1)
    axs[1].set_yticklabels(['{:.0f}%'.format(x * 100) for x in axs[1].get_yticks()])

    # Sorting the legend handles
    handles, labels = axs[1].get_legend_handles_labels()
    allocation_type_totals = allocation_over_time.sum().to_dict()
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: allocation_type_totals[x[1]], reverse=True)
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    sorted_labels = [label.capitalize() for label in sorted_labels]
    axs[1].legend(sorted_handles, sorted_labels, fontsize=12, loc='upper left')

    axs[1].grid(linestyle='--', alpha=0.6, zorder=1)
    axs[1].tick_params(axis='x', labelsize=14)
    axs[1].tick_params(axis='y', labelsize=13)
    axs[1].text(-0.02, 1.09, 'B', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # Adjust layout
    plt.tight_layout()

    # Save the plot to a local folder
    plt.savefig(output_file)


def viz_masking(trial_design, output_file):
    # Prepare masking data
    unique_pairs_masking = trial_design[['nct_id', 'masking', 'start_year']].drop_duplicates()
    masking_counts = unique_pairs_masking['masking'].value_counts().sort_values(ascending=True)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot for Masking Frequency (Plot A)
    bars_0 = axs[0].barh(masking_counts.index, masking_counts, color='lightgrey', zorder=2)
    for bar in bars_0:
        width = bar.get_width()
        axs[0].text(width, bar.get_y() + bar.get_height() / 2, f'{width}', va='center', fontsize=12)
    axs[0].tick_params(axis='y', labelsize=14)
    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].grid(axis='x', linestyle='--', alpha=0.6, zorder=1)
    axs[0].set_xlabel('Count of Unique Trials', fontsize=14)
    axs[0].set_title('Masking', fontsize=14)
    axs[0].set_xlim(0, max(masking_counts) + 630)  # Adjusted to max count for relevancy
    axs[0].text(-0.02, 1.08, 'A', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # Plot for Change in Masking Over Time (Plot B)

    # Prepare masking over time data
    masking_over_time = unique_pairs_masking.groupby(['start_year', 'masking']).size().unstack(fill_value=0)

    # Calculate the proportion of each masking type from all trials for each year
    masking_proportion_over_time = masking_over_time.div(masking_over_time.sum(axis=1), axis=0)
    masking_types = ['not reported', 'None (Open Label)', 'Single', 'Double', 'Triple', 'Quadruple']

    masking_colors = {'None (Open Label)': '#88CCEE',
    'Quadruple': '#CC6677',
    'not reported': '#DDCC77',
    'Double': '#117733',
    'Triple': '#AA4499',
    'Single': '#44AA99'}

    allocation_types = ['Non-Randomized', 'not reported', 'Randomized']

    bottom = np.zeros(len(masking_proportion_over_time))
    for masking_type in masking_types:
        bars = axs[1].bar(masking_proportion_over_time.index, 
                        masking_proportion_over_time[masking_type], 
                        bottom=bottom, 
                        label=masking_type, color=masking_colors.get(masking_type, 'gray'), zorder=2)
        bottom += masking_proportion_over_time[masking_type]

        # Add labels to each segment
        for bar in bars:
            height = bar.get_height()
            #print(height)
            if height > 0.055:
                axs[1].text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{height:.0%}', 
                            ha='center', va='center', fontsize=10, rotation=90) 

    axs[1].set_xlabel('Trial Start Year', fontsize=14)
    axs[1].set_title('Proportion of Reported Masking Over Time', fontsize=14)

    # Set y-axis limits and labels
    axs[1].set_ylim(0, 1)
    axs[1].set_yticklabels(['{:.0f}%'.format(x * 100) for x in axs[1].get_yticks()])

    # Sort and place the legend outside the plot area
    handles, labels = axs[1].get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: masking_types[::-1].index(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    axs[1].legend(sorted_handles, [label.capitalize() for label in sorted_labels], fontsize=12, loc='upper left', bbox_to_anchor=(1.00, 1))

    axs[1].grid(linestyle='--', alpha=0.6, zorder=1)
    axs[1].tick_params(axis='x', labelsize=14)
    axs[1].tick_params(axis='y', labelsize=13)
    axs[1].text(-0.02, 1.08, 'B', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # Adjust layout
    plt.tight_layout()

    # Save the plot to a local folder
    plt.savefig(output_file)

def viz_facilities_enrollment(trial_design, trials_with_participants, output_file):
    labels = ['1', '2-10', '11-20', '21-30', '31-40', '41-50', '>50']

    # Prepare masking data
    unique_pairs_facilities = trial_design[['nct_id', 'binned_facilities', 'number_of_facilities']].drop_duplicates()

    # Count occurrences in each bin including NaN for 'not reported'
    bin_counts_facilities = unique_pairs_facilities['binned_facilities'].value_counts().reindex(labels + [np.nan]).fillna(0)
    bin_counts_facilities[np.nan] = len(unique_pairs_facilities[unique_pairs_facilities['number_of_facilities'].isna()])

    # Count the occurrences of each enrollment class
    enrollment_class_counts = trials_with_participants['enrollment_class'].value_counts().sort_index()

    # Define the order of the categories
    category_order_enrollment = ["0–10", "11–50", "51–100", "101–1,000", ">1,000"]

    # Reorder the counts according to the specified category order
    enrollment_class_counts = enrollment_class_counts.reindex(category_order_enrollment, fill_value=0)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))

    # Plot for number_of_facilities
    bin_counts_facilities = bin_counts_facilities.rename(index={np.nan: 'N.A.'})
    bars_facilities = axs[0].bar(bin_counts_facilities.index.astype(str), bin_counts_facilities, color='lightgrey', zorder=3)
    for i, value in enumerate(bin_counts_facilities):
        axs[0].text(i, value + 0.1, str(int(value)), ha='center', va='bottom')

    axs[0].tick_params(axis='y', labelsize=14)
    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].grid(axis='y', linestyle='--', alpha=0.6, zorder=1)
    axs[0].set_xlabel('Number of Facilities', fontsize=14)
    axs[0].set_title('Number of Facilities', fontsize=14)
    #plt.xticks(rotation=45)
    axs[0].text(-0.01, 1.08, 'A', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # Plot for enrollment
    bars_enrollment = axs[1].bar(enrollment_class_counts.index, enrollment_class_counts.values, color='lightgrey', zorder=3)
    for i, value in enumerate(enrollment_class_counts):
        axs[1].text(i, value + 0.1, str(int(value)), ha='center', va='bottom')

    axs[1].tick_params(axis='y', labelsize=14)
    axs[1].tick_params(axis='x', labelsize=14)
    axs[1].grid(axis='y', linestyle='--', alpha=0.6, zorder=1)
    axs[1].set_xlabel('Number of Participants (Actual or Anticipated)', fontsize=14)
    axs[1].set_title('Number of Enrolled Participants', fontsize=14)
    axs[1].text(-0.01, 1.08, 'B', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    axs[1].set_ylim(0, max(enrollment_class_counts) + 1000) # Adjusted to max count for relevancy

    plt.tight_layout()

    plt.savefig(output_file)

def viz_outcomes(trial_design, output_file):
    labels = ['1', '2-5', '6-10', '11-20', '>20']

    unique_pairs_primary_outocome = trial_design[['nct_id', 'number_of_primary_outcomes_to_measure', 'binned_primary_outcomes']].drop_duplicates()

    # Count occurrences in each bin including NaN for 'not reported'
    bin_counts_primary_outcomes = unique_pairs_primary_outocome['binned_primary_outcomes'].value_counts().reindex(labels + [np.nan]).fillna(0)
    bin_counts_primary_outcomes[np.nan] = len(unique_pairs_primary_outocome[unique_pairs_primary_outocome['number_of_primary_outcomes_to_measure'].isna()])

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))

    bin_counts_primary_outcomes = bin_counts_primary_outcomes.rename(index={pd.NA: 'N.A.'})
    bin_counts_primary_outcomes.index = bin_counts_primary_outcomes.index.fillna('N.A.').astype(str)
    # Plot A
    bars_0 = axs[0].bar(bin_counts_primary_outcomes.index, bin_counts_primary_outcomes, color='lightgrey', zorder=2)

    for i, value in enumerate(bin_counts_primary_outcomes):
        axs[0].text(i, value + 0.1, str(int(value)), ha='center', va='bottom')
        
    axs[0].tick_params(axis='y', labelsize=14)
    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].grid(axis='y', linestyle='--', alpha=0.6, zorder=1)
    axs[0].set_xlabel('Number of Primary Outcomes', fontsize=14)
    axs[0].set_title('Primary Outcomes', fontsize=14)
    axs[0].text(-0.03, 1.05, 'A', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # Plot B
    unique_pairs_secondary_outocome = trial_design[['nct_id', 'number_of_secondary_outcomes_to_measure', 'binned_secondary_outcomes']].drop_duplicates()

    # Count occurrences in each bin including NaN for 'not reported'
    bin_counts_secondary_outcomes = unique_pairs_secondary_outocome['binned_secondary_outcomes'].value_counts().reindex(labels + [np.nan]).fillna(0)
    bin_counts_secondary_outcomes[np.nan] = len(unique_pairs_secondary_outocome[unique_pairs_secondary_outocome['number_of_secondary_outcomes_to_measure'].isna()])
    bin_counts_secondary_outcomes = bin_counts_secondary_outcomes.rename(index={pd.NA: 'N.A.'})
    bin_counts_secondary_outcomes.index = bin_counts_secondary_outcomes.index.fillna('N.A.').astype(str)

    bars_1 = axs[1].bar(bin_counts_secondary_outcomes.index, bin_counts_secondary_outcomes, color='lightgrey', zorder=2)

    for i, value in enumerate(bin_counts_secondary_outcomes):
        axs[1].text(i, value + 0.1, str(int(value)), ha='center', va='bottom')
        
    axs[1].tick_params(axis='y', labelsize=14)
    axs[1].tick_params(axis='x', labelsize=14)
    axs[1].grid(axis='y', linestyle='--', alpha=0.6, zorder=1)
    axs[1].set_xlabel('Number of Secondary Outcomes', fontsize=14)
    axs[1].set_title('Secondary Outcomes', fontsize=14)
    axs[1].text(-0.03, 1.05, 'B', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    plt.tight_layout()

    plt.savefig(output_file)

def viz_results_reporting(trial_metadata, output_file):

    df_results_reported = trial_metadata[trial_metadata['overall_status']=='Completed']
    df_results_reported = df_results_reported[df_results_reported['completion_year']<2022]

    year_to_use = 'completion_year'

    # Group by completion_year and were_results_reported, then count unique nct_ids
    results_over_time = df_results_reported.groupby([year_to_use, 'were_results_reported'])['nct_id'].nunique().unstack(fill_value=0)

    # Calculate the proportion of each type
    results_proportion_over_time = results_over_time.div(results_over_time.sum(axis=1), axis=0)

    # Filter data for reported results
    reported_results = df_results_reported[df_results_reported['were_results_reported'] == True]

    # Calculate average months to report results and standard deviation
    average_months_to_report = reported_results.groupby(year_to_use)['months_to_report_results'].mean()
    std_months_to_report = reported_results.groupby(year_to_use)['months_to_report_results'].std()

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Subplot 1: Stacked bar chart for results reported/results pending
    bottom = np.zeros(len(results_proportion_over_time))
    colors = {False: '#DDCC77', True: '#44AA99'}
    for reported in [False, True]:
        label = 'Results Reported' if reported else 'Results Pending'
        bars = ax1.bar(results_proportion_over_time.index, 
                    results_proportion_over_time[reported], 
                    bottom=bottom, 
                    label=label, 
                    color=colors[reported], zorder=2)
        bottom += results_proportion_over_time[reported]

        # Add labels to each segment
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2, f'{height*100:.0f}%', 
                        ha='center', va='center', fontsize=10, rotation=90)

    ax1.set_xlabel("Trial " + year_to_use.replace("_", " ").capitalize(), fontsize=14)
    #ax1.set_ylabel('Proportion of Trials', fontsize=14)
    ax1.set_title('Proportion Reported Results for Trials Completed in that Year', fontsize=15)
    ax1.legend(loc='lower left', title='', fontsize=13)
    ax1.grid(linestyle='--', alpha=0.6, zorder=1)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_ylim(0, 1)
    ax1.set_yticklabels(['{:.0f}%'.format(x * 100) for x in ax1.get_yticks()])
    ax1.text(-0.03, 1.08, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')


    # Subplot 2: Line chart for months to report results
    ax2.plot(average_months_to_report.index, average_months_to_report, marker='o', label='Avg Months to Report Results', color='#882255')
    ax2.fill_between(average_months_to_report.index, average_months_to_report - std_months_to_report, average_months_to_report + std_months_to_report, color='lightgrey', alpha=0.3)
    ax2.set_xlabel("Trial " + year_to_use.replace("_", " ").capitalize(), fontsize=14)
    #ax2.set_ylabel('Average Months to Report Results', fontsize=14)
    ax2.set_title('Average Number of Months to Report Results for Trials Completed in that Year', fontsize=15)
    #ax2.legend(loc='upper right', title='', fontsize=13)
    ax2.grid(linestyle='--', alpha=0.6, zorder=1)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.text(-0.03, 1.08, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    plt.tight_layout()

    # Save the figure as a PDF
    plt.savefig(output_file)
    
def viz_sponsorship_lead(trial_metadata, output_file):
    df_filtered = trial_metadata[trial_metadata['lead_or_collaborator']=='lead']
    df_filtered['start_year'] = df_filtered['start_year'].astype(int)

    # Define color map
    labels = ['UNKNOWN', 'INDIV', 'NETWORK', 'OTHER_GOV', 'FED', 'NIH', 'HOSPITAL', 'OTHER', 'UNIVERSITY', 'INDUSTRY']
    colors = ['#d3d3d3', '#000000', '#999999', '#56B4E9', '#D55E00', '#F0E442', '#CC79A7', '#009E73', '#0072B2', '#E69F00']
    color_map = dict(zip(labels, colors))

    default_color = '#FFFFFF'  # White, change as needed

    # Creating a pivot table with counts per agency_class and year
    pivot_table_counts = df_filtered.pivot_table(index='start_year', columns='agency_class', values='nct_id', aggfunc='count', fill_value=0)
    pivot_table_percentage = pivot_table_counts.divide(pivot_table_counts.sum(axis=1), axis=0) * 100
    overall_distribution = pivot_table_counts.sum(axis=0).sort_values(ascending=True)

    # Reorder pivot table columns to match the order of overall_distribution
    pivot_table_counts = pivot_table_counts[overall_distribution.index]
    pivot_table_percentage = pivot_table_percentage[overall_distribution.index]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1 - Overall Distribution (now plot A)
    overall_distribution.plot(kind='barh', ax=ax1, color=[color_map.get(x, default_color) for x in overall_distribution.index], zorder=2)
    ax1.set_title('Overall Distribution of Lead Funding Sources since year 2000', fontsize=14)
    ax1.set_xlabel('Count of Unique Trials', fontsize=14)
    ax1.text(-0.03, 1.05, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    ax1.set_ylabel('')
    ax1.grid(axis='x', linestyle='--', alpha=0.6, zorder=1)

    # Adding text labels to the bars
    for index, value in enumerate(overall_distribution):
        ax1.text(value, index, f' {int(value)}', va='center', ha='left')
    ax1.set_xlim(0, max(overall_distribution) + 500)
    ax1.tick_params(axis='both', labelsize=13)  # Increase tick label size

    # Plot 2 - Yearly Distribution (now plot B)
    pivot_table_percentage.plot(kind='bar', stacked=True, ax=ax2, color=[color_map.get(x, default_color) for x in pivot_table_counts.columns])
    ax2.set_title('Percentage of Total Trials by Lead Funding Source since year 2000', fontsize=14)
    ax2.set_xlabel('Trial Start Year', fontsize=14)
    ax2.legend().set_visible(False)
    ax2.tick_params(axis='both', labelsize=13)  # Increase tick label size

    ax2.text(-0.03, 1.05, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    # Adding text inside the bars
    for bars_stack in ax2.containers:
        ax2.bar_label(bars_stack, labels=[f'{v:.0f}%' if v > 10 else '' for v in bars_stack.datavalues], label_type='center', fontsize=8)

    # Synchronize legend for both plots
    handles, labels = ax2.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1], title='Funding Source Class', loc='lower right')

    plt.tight_layout()

    # Save the figure as a PDF
    plt.savefig(output_file)