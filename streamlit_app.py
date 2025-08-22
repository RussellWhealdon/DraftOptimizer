# streamlit_app.py
import pandas as pd
import streamlit as st
from engine import (
	default_settings,
	normalize_players,
	default_draft_log,
	draft_player,
	compute_availability,
)

st.set_page_config(page_title="Draft Availability Optimizer", layout="wide")

@st.cache_data
def load_players() -> pd.DataFrame:
	path = "HHL Pre Post Analysis - Player Data Export.csv"
	df = pd.read_csv(path)
	return normalize_players(df)

def init_state():
	if "players_df" not in st.session_state:
		st.session_state.players_df = load_players()
	if "draft_log_df" not in st.session_state:
		st.session_state.draft_log_df = default_draft_log()
	if "settings" not in st.session_state:
		st.session_state.settings = default_settings.copy()
	if "my_team_slot" not in st.session_state:
		st.session_state.my_team_slot = 8
	if "top_n" not in st.session_state:
		st.session_state.top_n = 30

init_state()

st.title("Fantasy Draft Availability Engine (MVP)")
with st.sidebar:
	st.session_state.my_team_slot = st.number_input("My team slot", min_value=1, max_value=st.session_state.settings["num_teams"], value=st.session_state.my_team_slot, step=1)
	st.session_state.top_n = st.slider("Top N candidates", min_value=10, max_value=60, value=st.session_state.top_n, step=5)

with st.expander("Draft Log"):
    # Ensure normalized players (has 'name')
    players_df = st.session_state.players_df.copy()
    players_df["player_id"] = players_df["player_id"].astype(str)

    # Prepare draft log for join
    dl = st.session_state.draft_log_df.copy()
    # Keep None as-is so they show as blank; only cast non-null IDs for matching
    dl_ids = dl["player_id"].dropna().astype(str)
    dl.loc[dl["player_id"].notna(), "player_id"] = dl_ids

    # Join to get player names/pos
    show_df = dl.merge(
        players_df[["player_id", "name", "pos"]],
        on="player_id",
        how="left"
    )

    # Order and tidy columns
    show_df = show_df[["overall_pick", "round", "team_slot", "player_id", "name", "pos"]]
    show_df = show_df.rename(columns={"name": "player_name", "pos": "player_pos"})

    st.dataframe(show_df, use_container_width=True, hide_index=True)

# Add My Drafted Team expander
with st.expander("My Drafted Team"):
    # Get all picks for my team
    my_picks = st.session_state.draft_log_df[
        (st.session_state.draft_log_df["team_slot"] == st.session_state.my_team_slot) & 
        (st.session_state.draft_log_df["player_id"].notna())
    ].copy()
    
    if my_picks.empty:
        st.info("You haven't drafted any players yet.")
    else:
        # Join with player info to get names and positions
        players_df = st.session_state.players_df.copy()
        players_df["player_id"] = players_df["player_id"].astype(str)
        
        my_team_df = my_picks.merge(
            players_df[["player_id", "name", "pos", "tier", "adp_overall", "proj_points"]],
            on="player_id",
            how="left"
        )
        
        # Reorder columns and rename for display
        my_team_df = my_team_df[["overall_pick", "round", "name", "pos", "tier", "adp_overall", "proj_points"]]
        my_team_df = my_team_df.rename(columns={
            "name": "Player Name",
            "pos": "Position", 
            "tier": "Tier",
            "adp_overall": "ADP",
            "proj_points": "Proj Points"
        })
        
        # Sort by overall pick (draft order)
        my_team_df = my_team_df.sort_values("overall_pick")
        
        # Display the team
        st.dataframe(my_team_df, use_container_width=True, hide_index=True)
        
        # Show summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Players", len(my_team_df))
        
        with col2:
            pos_counts = my_team_df["Position"].value_counts()
            st.metric("Positions Filled", len(pos_counts))
        
        with col3:
            if "ADP" in my_team_df.columns and not my_team_df["ADP"].isna().all():
                avg_adp = my_team_df["ADP"].mean()
                st.metric("Avg ADP", f"{avg_adp:.1f}")
            else:
                st.metric("Avg ADP", "N/A")
        
        with col4:
            if "Proj Points" in my_team_df.columns and not my_team_df["Proj Points"].isna().all():
                total_proj = my_team_df["Proj Points"].sum()
                st.metric("Total Proj Points", f"{total_proj:.0f}")
            else:
                st.metric("Total Proj Points", "N/A")

avail_df = compute_availability(
	players_df=st.session_state.players_df,
	draft_log_df=st.session_state.draft_log_df,
	settings=st.session_state.settings,
	my_team_slot=st.session_state.my_team_slot,
	top_n=st.session_state.top_n,
)

st.subheader("Available Players and Survival Probability")
if avail_df.empty:
	st.info("No available players or next pick not found.")
else:
	cols = st.columns([2,1,1,1,1,1,1,1,1])
	cols[0].markdown("**Name**")
	cols[1].markdown("**Pos**")
	cols[2].markdown("**Tier**")
	cols[3].markdown("**ADP**")
	cols[4].markdown("**Proj**")
	cols[5].markdown("**P(survive)**")
	cols[8].markdown("**Action**")
	for _, row in avail_df.iterrows():
		c = st.columns([2,1,1,1,1,1,1,1,1])
		c[0].write(f"{row['Name']}")
		c[1].write(f"{row['Pos']}")
		c[2].write(f"{row['Tier']}")
		c[3].write(f"{row['ADP']:.0f}" if pd.notna(row["ADP"]) else "-")
		c[4].write(f"{row['Proj_Points']:.1f}" if pd.notna(row["Proj_Points"]) else "-")
		c[5].write(f"{row['P_survive_to_next_pick']*100:.2f}%")
		if c[8].button("Draft", key=f"draft_{row['player_id']}"):
			st.session_state.draft_log_df = draft_player(st.session_state.draft_log_df, str(row["player_id"]))
			st.rerun()
