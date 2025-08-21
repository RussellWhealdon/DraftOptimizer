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
