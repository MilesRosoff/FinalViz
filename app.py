import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import plotly.express as px

# 1. Page Configuration & CSS Padding Fix
st.set_page_config(page_title="Situational Play-Caller", layout="wide")

st.markdown("""
    <style>
           .block-container {
                padding-top: 1.5rem;
                padding-bottom: 1rem;
            }
    </style>
    """, unsafe_allow_html=True)

st.title("Situational Play-Caller Dashboard")
st.markdown("Analyzing historical NFL play success rates to drive smarter decision-making.")

# NFL Team Colors Dictionary
NFL_COLORS = {
    'ARI': ['#97233F', '#000000'], 'ATL': ['#A71930', '#000000'], 'BAL': ['#241773', '#9E7C0C'],
    'BUF': ['#00338D', '#C60C30'], 'CAR': ['#0085CA', '#101820'], 'CHI': ['#0B162A', '#C83803'],
    'CIN': ['#FB4F14', '#000000'], 'CLE': ['#311D00', '#FF3C00'], 'DAL': ['#003594', '#869397'],
    'DEN': ['#FB4F14', '#002244'], 'DET': ['#0076B6', '#B0B7BC'], 'GB': ['#203731', '#FFB612'],
    'HOU': ['#03202F', '#A71930'], 'IND': ['#002C5F', '#A2AAAD'], 'JAX': ['#101820', '#D7A22A'],
    'KC': ['#E31837', '#FFB81C'], 'LV':  ['#000000', '#A5ACAF'], 'LAC': ['#0080C6', '#FFC20E'],
    'LAR': ['#003594', '#FFA300'], 'MIA': ['#008E97', '#FC4C02'], 'MIN': ['#4F2683', '#FFC62F'],
    'NE':  ['#002244', '#C60C30'], 'NO':  ['#D3BC8D', '#101820'], 'NYG': ['#0B2265', '#A71930'],
    'NYJ': ['#125740', '#000000'], 'PHI': ['#004C54', '#A5ACAF'], 'PIT': ['#101820', '#FFB612'],
    'SF':  ['#AA0000', '#B3995D'], 'SEA': ['#002244', '#69BE28'], 'TB':  ['#D50A0A', '#FF7900'],
    'TEN': ['#0C2340', '#4B92DB'], 'WAS': ['#5A1414', '#FFB612']
}

# 2. Data Loading & Preprocessing
@st.cache_data
def load_and_clean_data():
    years = [2025]
    with st.spinner('Fetching NFL play-by-play data...'):
        df = nfl.import_pbp_data(years)
    
    df = df[df['play_type'].isin(['run', 'pass'])]
    cols_to_keep = ['down', 'ydstogo', 'yardline_100', 'play_type', 'epa', 'posteam', 'defteam']
    df = df[cols_to_keep]
    df = df.dropna(subset=['down', 'ydstogo', 'epa', 'posteam', 'defteam'])
    
    df['is_success'] = df['epa'] > 0
    return df

df = load_and_clean_data()

# 3. Sidebar Filters
st.sidebar.header("Game Situation Filters")

team_list = ["All Teams"] + sorted(df['posteam'].unique().tolist())
selected_team = st.sidebar.selectbox("Select Offense Team", options=team_list)

def_team_list = ["All Teams"] + sorted(df['defteam'].unique().tolist())
selected_def_team = st.sidebar.selectbox("Select Defense Team", options=def_team_list)

selected_down = st.sidebar.selectbox(
    "Select Down",
    options=["All Downs", 1.0, 2.0, 3.0, 4.0],
    format_func=lambda x: x if x == "All Downs" else f"{int(x)}{'st' if x==1 else 'nd' if x==2 else 'rd' if x==3 else 'th'} Down"
)

min_yds = int(df['ydstogo'].min())
max_yds = int(df['ydstogo'].max())
selected_distance = st.sidebar.slider(
    "Yards to Go Range", 
    min_value=min_yds, 
    max_value=max_yds, 
    value=(1, 10)
)

# Apply filters
filtered_df = df.copy()

if selected_team != "All Teams":
    filtered_df = filtered_df[filtered_df['posteam'] == selected_team]
if selected_def_team != "All Teams":
    filtered_df = filtered_df[filtered_df['defteam'] == selected_def_team]
if selected_down != "All Downs":
    filtered_df = filtered_df[filtered_df['down'] == selected_down]

filtered_df = filtered_df[(filtered_df['ydstogo'] >= selected_distance[0]) & (filtered_df['ydstogo'] <= selected_distance[1])]

# Determine Chart Colors
if selected_team == "All Teams" or selected_team not in NFL_COLORS:
    dynamic_colors = {'pass': '#013369', 'run': '#D50A0A'}
else:
    dynamic_colors = {'pass': NFL_COLORS[selected_team][0], 'run': NFL_COLORS[selected_team][1]}

# 4. AI Recommendation Engine
if not filtered_df.empty:
    run_df = filtered_df[filtered_df['play_type'] == 'run']
    pass_df = filtered_df[filtered_df['play_type'] == 'pass']
    
    run_epa = run_df['epa'].mean() if not run_df.empty else 0
    pass_epa = pass_df['epa'].mean() if not pass_df.empty else 0
    
    if pass_epa > run_epa:
        recommendation = "PASS"
        best_epa = pass_epa
    else:
        recommendation = "RUN"
        best_epa = run_epa
        
    st.success(f"**AI Recommendation: {recommendation}** (Expected EPA: {best_epa:.3f}) | Average Pass EPA: {pass_epa:.3f} | Average Run EPA: {run_epa:.3f}")

# 5. Visualizations
if filtered_df.empty:
    st.warning("No data available for this specific situation. Try adjusting the filters.")
else:
    col1, col2 = st.columns(2)

    with col1:
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
            color_discrete_map=dynamic_colors
        )
        
        fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        
        # Strip margins and lock height to prevent scrolling
        if not success_rates.empty:
            y_min = success_rates['is_success'].min() * 0.8
            y_max = min(success_rates['is_success'].max() * 1.2, 105) 
            fig1.update_layout(
                yaxis_range=[max(0, y_min), y_max],
                margin=dict(t=20, b=20, l=20, r=20),
                height=350
            )
        
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Play Distribution by Field Position")
        
        fig2 = px.histogram(
            filtered_df, 
            x='yardline_100', 
            color='play_type',
            nbins=10,
            labels={'yardline_100': 'Yards from Endzone (100 = Own Goal Line)', 'count': 'Number of Plays'},
            barmode='group',
            color_discrete_map=dynamic_colors
        )
        
        # Strip margins and lock height to match fig1
        fig2.update_layout(
            xaxis=dict(autorange="reversed", showgrid=True, gridcolor='white', gridwidth=2, dtick=10),
            yaxis=dict(showgrid=False),
            plot_bgcolor='#2ca02c', 
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=20, l=20, r=20),
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)