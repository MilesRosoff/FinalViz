import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Situational Play-Caller", layout="wide")
st.title("Situational Play-Caller Dashboard")
st.markdown("Analyzing historical NFL play success rates to drive smarter decision-making.")

# 2. Data Loading & Preprocessing
@st.cache_data
def load_and_clean_data():
    years = [2025]
    with st.spinner('Fetching NFL play-by-play data...'):
        df = nfl.import_pbp_data(years)
    
    # Filter out punts/penalties
    df = df[df['play_type'].isin(['run', 'pass'])]
    
    # Added 'posteam' to our kept columns for the team filter
    cols_to_keep = ['down', 'ydstogo', 'yardline_100', 'play_type', 'epa', 'posteam']
    df = df[cols_to_keep]
    df = df.dropna(subset=['down', 'ydstogo', 'epa', 'posteam'])
    
    # Feature Engineering
    df['is_success'] = df['epa'] > 0
    
    # Categorize distance for easier filtering
    def categorize_distance(yards):
        if yards <= 3:
            return 'Short (1-3 yds)'
        elif yards <= 7:
            return 'Medium (4-7 yds)'
        else:
            return 'Long (8+ yds)'
            
    df['distance_category'] = df['ydstogo'].apply(categorize_distance)
    return df

df = load_and_clean_data()

# 3. Sidebar Filters
st.sidebar.header("Game Situation Filters")

# Extract unique teams for the dropdown and add "All Teams"
team_list = ["All Teams"] + sorted(df['posteam'].unique().tolist())
selected_team = st.sidebar.selectbox("Select Team (Offense)", options=team_list)

selected_down = st.sidebar.selectbox(
    "Select Down",
    options=["All Downs", 1.0, 2.0, 3.0, 4.0],
    format_func=lambda x: x if x == "All Downs" else f"{int(x)}{'st' if x==1 else 'nd' if x==2 else 'rd' if x==3 else 'th'} Down"
)

selected_distance = st.sidebar.selectbox(
    "Select Distance",
    options=["All Distances", 'Short (1-3 yds)', 'Medium (4-7 yds)', 'Long (8+ yds)']
)

# Apply filters dynamically based on user selection
filtered_df = df.copy()

if selected_team != "All Teams":
    filtered_df = filtered_df[filtered_df['posteam'] == selected_team]

if selected_down != "All Downs":
    filtered_df = filtered_df[filtered_df['down'] == selected_down]

if selected_distance != "All Distances":
    filtered_df = filtered_df[filtered_df['distance_category'] == selected_distance]

# Dynamic Subtitle to show what the user is looking at
down_text = selected_down if selected_down == 'All Downs' else f'{int(selected_down)} Down'
st.write(f"### Analyzing: {selected_team} | {down_text} | {selected_distance}")

# 4. Visualizations
if filtered_df.empty:
    st.warning("No data available for this specific situation. Try adjusting the filters.")
else:
    col1, col2 = st.columns(2)

    with col1:
        # Visualization 1: Run vs Pass Efficiency Matrix
        st.subheader("Run vs. Pass Success Rate")
        
        success_rates = filtered_df.groupby('play_type')['is_success'].mean().reset_index()
        success_rates['is_success'] = success_rates['is_success'] * 100 
        
        fig1 = px.bar(
            success_rates, 
            x='play_type', 
            y='is_success',
            text='is_success', 
            labels={'play_type': 'Play Type', 'is_success': 'Success Rate (%)'},
            color='play_type',
            color_discrete_map={'pass': '#1f77b4', 'run': '#ff7f0e'}
        )
        
        fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        
        # Safely calculate limits to ensure the chart scales beautifully
        if not success_rates.empty:
            y_min = success_rates['is_success'].min() * 0.8
            y_max = min(success_rates['is_success'].max() * 1.2, 105) 
            fig1.update_layout(yaxis_range=[max(0, y_min), y_max])
        
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Visualization 2: The Situational Field Map Heatmap
        st.subheader("Success Heatmap by Field Position")
        
        # Break the 100 yard field into 10-yard zones
        bins = list(range(0, 101, 10))
        labels = [f'{i}-{i+10}' for i in range(0, 100, 10)]
        
        filtered_df = filtered_df.copy()
        filtered_df['field_zone'] = pd.cut(filtered_df['yardline_100'], bins=bins, labels=labels, right=False)
        
        # Calculate success rate by zone
        heatmap_data = filtered_df.groupby(['play_type', 'field_zone'], observed=True)['is_success'].mean().reset_index()
        heatmap_data['is_success'] = heatmap_data['is_success'] * 100
        
        # Pivot for the heatmap format
        heatmap_pivot = heatmap_data.pivot(index='play_type', columns='field_zone', values='is_success')
        
        fig2 = px.imshow(
            heatmap_pivot,
            labels=dict(x="Yards from Endzone (100 = Own Goal Line)", y="Play Type", color="Success Rate (%)"),
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            color_continuous_scale="Greens",
            text_auto='.1f',
            aspect="auto"
        )
        # Reverse X axis so 100 is on the left
        fig2.update_layout(xaxis=dict(autorange="reversed")) 
        st.plotly_chart(fig2, use_container_width=True)