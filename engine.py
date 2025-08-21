# engine.py
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

default_settings: Dict = {
	"num_teams": 10,
	"num_rounds": 15,
	"positions": ["QB", "RB", "WR", "TE", "K", "DEF"],
	"roster_max": {"QB": 3, "RB": 6, "WR": 7, "TE": 3, "K": 2, "DEF": 2},
	"starters_req": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1},
	"flex_positions": ["RB", "WR", "TE"],
	"needs_weights": {"alpha_starter": 1.0, "beta_bench": 0.5, "gamma_tier": 0.5, "delta_market": 0.25},
}

def normalize_players(players_df: pd.DataFrame) -> pd.DataFrame:
	df = players_df.copy()
	rename_map = {
		"Name": "name",
		"Pos": "pos",
		"Team": "team_nfl",
		"ADP": "adp_overall",
		"Tier": "tier",
		"Point Proj": "proj_points",
		"ECR Rank": "ecr_rank",
		"BYE": "bye_week",
	}
	df = df.rename(columns=rename_map)
	if "player_id" not in df.columns:
		df["player_id"] = [str(1000000 + i) for i in range(len(df))]
	return df

# --- 2) Draft log DataFrame TEMPLATE ---
draft_log_cols = [
    ("overall_pick", "Int64"),  # 1..(num_teams * num_rounds)
    ("round", "Int64"),
    ("team_slot", "Int64"),     # 1..num_teams
    ("player_id", "string"),    # filled once picked; NaN for future picks
]

draft_log_template = pd.DataFrame({c: pd.Series(dtype=t) for c, t in draft_log_cols})

# Create a tiny demo for a 10-team snake draft, first ~2 rounds partially filled
demo_draft_log = pd.DataFrame([
    # round 1 (slots 1..10)
    {"overall_pick": 1, "round": 1, "team_slot": 1, "player_id": None},
    {"overall_pick": 2, "round": 1, "team_slot": 2, "player_id": None},
    {"overall_pick": 3, "round": 1, "team_slot": 3, "player_id": None},
    {"overall_pick": 4, "round": 1, "team_slot": 4, "player_id": None},
    {"overall_pick": 5, "round": 1, "team_slot": 5, "player_id": None},
    {"overall_pick": 6, "round": 1, "team_slot": 6, "player_id": None},
    {"overall_pick": 7, "round": 1, "team_slot": 7, "player_id": None},
    {"overall_pick": 8, "round": 1, "team_slot": 8, "player_id": None},
    {"overall_pick": 9, "round": 1, "team_slot": 9, "player_id": None},
    {"overall_pick": 10, "round": 1, "team_slot": 10, "player_id": None},
    # round 2 (snake: slots 10..1)
    {"overall_pick": 11, "round": 2, "team_slot": 10, "player_id": None},
    {"overall_pick": 12, "round": 2, "team_slot": 9, "player_id": None},
    {"overall_pick": 13, "round": 2, "team_slot": 8, "player_id": None},
    {"overall_pick": 14, "round": 2, "team_slot": 7, "player_id": None},
    {"overall_pick": 15, "round": 2, "team_slot": 6, "player_id": None},
    {"overall_pick": 16, "round": 2, "team_slot": 5, "player_id": None},
    {"overall_pick": 17, "round": 2, "team_slot": 4, "player_id": None},
    {"overall_pick": 18, "round": 2, "team_slot": 3, "player_id": None},
    {"overall_pick": 19, "round": 2, "team_slot": 2, "player_id": None},
    {"overall_pick": 20, "round": 2, "team_slot": 1, "player_id": None},
    # round 3 (snake: slots 1..10)
    {"overall_pick": 21, "round": 3, "team_slot": 1, "player_id": None},
    {"overall_pick": 22, "round": 3, "team_slot": 2, "player_id": None},
    {"overall_pick": 23, "round": 3, "team_slot": 3, "player_id": None},
    {"overall_pick": 24, "round": 3, "team_slot": 4, "player_id": None},
    {"overall_pick": 25, "round": 3, "team_slot": 5, "player_id": None},
    {"overall_pick": 26, "round": 3, "team_slot": 6, "player_id": None},
    {"overall_pick": 27, "round": 3, "team_slot": 7, "player_id": None},
    {"overall_pick": 28, "round": 3, "team_slot": 8, "player_id": None},
    {"overall_pick": 29, "round": 3, "team_slot": 9, "player_id": None},
    {"overall_pick": 30, "round": 3, "team_slot": 10, "player_id": None},
    # round 4 (snake: slots 10..1)
    {"overall_pick": 31, "round": 4, "team_slot": 10, "player_id": None},
    {"overall_pick": 32, "round": 4, "team_slot": 9, "player_id": None},
    {"overall_pick": 33, "round": 4, "team_slot": 8, "player_id": None},
    {"overall_pick": 34, "round": 4, "team_slot": 7, "player_id": None},
    {"overall_pick": 35, "round": 4, "team_slot": 6, "player_id": None},
    {"overall_pick": 36, "round": 4, "team_slot": 5, "player_id": "4014806"}, #Ben Keeper Allem 4th
    {"overall_pick": 37, "round": 4, "team_slot": 4, "player_id": None},
    {"overall_pick": 38, "round": 4, "team_slot": 3, "player_id": None},
    {"overall_pick": 39, "round": 4, "team_slot": 2, "player_id": None},
    {"overall_pick": 40, "round": 4, "team_slot": 1, "player_id": None},
    # round 5 (snake: slots 1..10)
    {"overall_pick": 41, "round": 5, "team_slot": 1, "player_id": "3473794"}, #Drew Keeper McBride 5th
    {"overall_pick": 42, "round": 5, "team_slot": 2, "player_id": None},
    {"overall_pick": 43, "round": 5, "team_slot": 3, "player_id": None},
    {"overall_pick": 44, "round": 5, "team_slot": 4, "player_id": "3313410"}, #Jaxon Keeper Kittle 5th
    {"overall_pick": 45, "round": 5, "team_slot": 5, "player_id": None},
    {"overall_pick": 46, "round": 5, "team_slot": 6, "player_id": "2793061"}, # Grayson Keeper Nabers 5th
    {"overall_pick": 47, "round": 5, "team_slot": 7, "player_id": None},
    {"overall_pick": 48, "round": 5, "team_slot": 8, "player_id": None},
    {"overall_pick": 49, "round": 5, "team_slot": 9, "player_id": None},
    {"overall_pick": 50, "round": 5, "team_slot": 10, "player_id": None},
    # round 6 (snake: slots 10..1)
    {"overall_pick": 51, "round": 6, "team_slot": 10, "player_id": None},
    {"overall_pick": 52, "round": 6, "team_slot": 9, "player_id": None},
    {"overall_pick": 53, "round": 6, "team_slot": 8, "player_id": None},
    {"overall_pick": 54, "round": 6, "team_slot": 7, "player_id": None},
    {"overall_pick": 55, "round": 6, "team_slot": 6, "player_id": None},
    {"overall_pick": 56, "round": 6, "team_slot": 5, "player_id": None},
    {"overall_pick": 57, "round": 6, "team_slot": 4, "player_id": None},
    {"overall_pick": 58, "round": 6, "team_slot": 3, "player_id": None},
    {"overall_pick": 59, "round": 6, "team_slot": 2, "player_id": None},
    {"overall_pick": 60, "round": 6, "team_slot": 1, "player_id": None},
    # round 7 (snake: slots 1..10)
    {"overall_pick": 61, "round": 7, "team_slot": 1, "player_id": None},
    {"overall_pick": 62, "round": 7, "team_slot": 2, "player_id": None},
    {"overall_pick": 63, "round": 7, "team_slot": 3, "player_id": None},
    {"overall_pick": 64, "round": 7, "team_slot": 4, "player_id": None},
    {"overall_pick": 65, "round": 7, "team_slot": 5, "player_id": None},
    {"overall_pick": 66, "round": 7, "team_slot": 6, "player_id": None},
    {"overall_pick": 67, "round": 7, "team_slot": 7, "player_id": None},
    {"overall_pick": 68, "round": 7, "team_slot": 8, "player_id": None},
    {"overall_pick": 69, "round": 7, "team_slot": 9, "player_id": None},
    {"overall_pick": 70, "round": 7, "team_slot": 10, "player_id": None},
    # round 8 (snake: slots 10..1)
    {"overall_pick": 71, "round": 8, "team_slot": 10, "player_id": None},
    {"overall_pick": 72, "round": 8, "team_slot": 9, "player_id": "7465604"}, # Steve Keeper J Daniels 8th
    {"overall_pick": 73, "round": 8, "team_slot": 8, "player_id": "6922949"}, # Russ Keeper NJigba 8th
    {"overall_pick": 74, "round": 8, "team_slot": 7, "player_id": None}, 
    {"overall_pick": 75, "round": 8, "team_slot": 6, "player_id": None},
    {"overall_pick": 76, "round": 8, "team_slot": 5, "player_id": None},
    {"overall_pick": 77, "round": 8, "team_slot": 4, "player_id": None},
    {"overall_pick": 78, "round": 8, "team_slot": 3, "player_id": None},
    {"overall_pick": 79, "round": 8, "team_slot": 2, "player_id": None},
    {"overall_pick": 80, "round": 8, "team_slot": 1, "player_id": None},
    # round 9 (snake: slots 1..10)
    {"overall_pick": 81, "round": 9, "team_slot": 1, "player_id": None},
    {"overall_pick": 82, "round": 9, "team_slot": 2, "player_id": None},
    {"overall_pick": 83, "round": 9, "team_slot": 3, "player_id": "6006621"}, # Rory Keeper Baker 9th
    {"overall_pick": 84, "round": 9, "team_slot": 4 ,"player_id": None},
    {"overall_pick": 85, "round": 9, "team_slot": 5, "player_id": None},
    {"overall_pick": 86, "round": 9, "team_slot": 6, "player_id": None},
    {"overall_pick": 87, "round": 9, "team_slot": 7, "player_id": None},
    {"overall_pick": 88, "round": 9, "team_slot": 8, "player_id": None},
    {"overall_pick": 89, "round": 9, "team_slot": 9, "player_id": None},
    {"overall_pick": 90, "round": 9, "team_slot": 10, "player_id": "4488657"}, # Bryson Keeper Irving 9th
    # round 10 (snake: slots 10..1)
    {"overall_pick": 91, "round": 10, "team_slot": 10, "player_id": None},
    {"overall_pick": 92, "round": 10, "team_slot": 9, "player_id": None},
    {"overall_pick": 93, "round": 10, "team_slot": 8, "player_id": None},
    {"overall_pick": 94, "round": 10, "team_slot": 7, "player_id": "4330861"}, # Daniel Keeper BTJ 10th
    {"overall_pick": 95, "round": 10, "team_slot": 6, "player_id": None},
    {"overall_pick": 96, "round": 10, "team_slot": 5, "player_id": None},
    {"overall_pick": 97, "round": 10, "team_slot": 4, "player_id": None},
    {"overall_pick": 98, "round": 10, "team_slot": 3, "player_id": None},
    {"overall_pick": 99, "round": 10, "team_slot": 2, "player_id": "1470696"}, # Alex Keeper Bowers 10th
    {"overall_pick": 100, "round": 10, "team_slot": 1, "player_id": None}, 
    # round 11 (snake: slots 1..10)
    {"overall_pick": 101, "round": 11, "team_slot": 1, "player_id": None},
    {"overall_pick": 102, "round": 11, "team_slot": 2, "player_id": None},
    {"overall_pick": 103, "round": 11, "team_slot": 3, "player_id": None},
    {"overall_pick": 104, "round": 11, "team_slot": 4, "player_id": None},
    {"overall_pick": 105, "round": 11, "team_slot": 5, "player_id": None},
    {"overall_pick": 106, "round": 11, "team_slot": 6, "player_id": None},
    {"overall_pick": 107, "round": 11, "team_slot": 7, "player_id": None},
    {"overall_pick": 108, "round": 11, "team_slot": 8, "player_id": None},
    {"overall_pick": 109, "round": 11, "team_slot": 9, "player_id": None},
    {"overall_pick": 110, "round": 11, "team_slot": 10, "player_id": None},
    # round 12 (snake: slots 10..1)
    {"overall_pick": 111, "round": 12, "team_slot": 10, "player_id": None},
    {"overall_pick": 112, "round": 12, "team_slot": 9, "player_id": None},
    {"overall_pick": 113, "round": 12, "team_slot": 8, "player_id": None},
    {"overall_pick": 114, "round": 12, "team_slot": 7, "player_id": None},
    {"overall_pick": 115, "round": 12, "team_slot": 6, "player_id": None},
    {"overall_pick": 116, "round": 12, "team_slot": 5, "player_id": None},
    {"overall_pick": 117, "round": 12, "team_slot": 4, "player_id": None},
    {"overall_pick": 118, "round": 12, "team_slot": 3, "player_id": None},
    {"overall_pick": 119, "round": 12, "team_slot": 2, "player_id": None},
    {"overall_pick": 120, "round": 12, "team_slot": 1, "player_id": None},
    # round 13 (snake: slots 1..10)
    {"overall_pick": 121, "round": 13, "team_slot": 1, "player_id": None},
    {"overall_pick": 122, "round": 13, "team_slot": 2, "player_id": None},
    {"overall_pick": 123, "round": 13, "team_slot": 3, "player_id": None},
    {"overall_pick": 124, "round": 13, "team_slot": 4, "player_id": None},
    {"overall_pick": 125, "round": 13, "team_slot": 5, "player_id": None},
    {"overall_pick": 126, "round": 13, "team_slot": 6, "player_id": None},
    {"overall_pick": 127, "round": 13, "team_slot": 7, "player_id": None},
    {"overall_pick": 128, "round": 13, "team_slot": 8, "player_id": None},
    {"overall_pick": 129, "round": 13, "team_slot": 9, "player_id": None},
    {"overall_pick": 130, "round": 13, "team_slot": 10, "player_id": None},
    # round 14 (snake: slots 10..1)
    {"overall_pick": 131, "round": 14, "team_slot": 10, "player_id": None},
    {"overall_pick": 132, "round": 14, "team_slot": 9, "player_id": None},
    {"overall_pick": 133, "round": 14, "team_slot": 8, "player_id": None},
    {"overall_pick": 134, "round": 14, "team_slot": 7, "player_id": None},
    {"overall_pick": 135, "round": 14, "team_slot": 6, "player_id": None},
    {"overall_pick": 136, "round": 14, "team_slot": 5, "player_id": None},
    {"overall_pick": 137, "round": 14, "team_slot": 4, "player_id": None},
    {"overall_pick": 138, "round": 14, "team_slot": 3, "player_id": None},
    {"overall_pick": 139, "round": 14, "team_slot": 2, "player_id": None},
    {"overall_pick": 140, "round": 14, "team_slot": 1, "player_id": None},
])

draft_log_df = pd.concat([draft_log_template, demo_draft_log], ignore_index=True)
	# Fill the rest of the grid (None player_id) up to num_teams*num_rounds
	df = pd.DataFrame(rows)
	num_teams = default_settings["num_teams"]
	num_rounds = default_settings["num_rounds"]
	max_picks = num_teams * num_rounds
	if df["overall_pick"].max() < max_picks:
		all_rows = []
		for pick in range(1, max_picks + 1):
			if pick in df["overall_pick"].values:
				all_rows.append(df[df["overall_pick"] == pick].iloc[0].to_dict())
			else:
				r = (pick - 1) // num_teams + 1
				if r % 2 == 1:
					team_slot = pick - (r - 1) * num_teams
				else:
					team_slot = num_teams - (pick - (r - 1) * num_teams) + 1
				all_rows.append({"overall_pick": pick, "round": r, "team_slot": team_slot, "player_id": None})
		df = pd.DataFrame(all_rows)
	return df[["overall_pick", "round", "team_slot", "player_id"]].sort_values("overall_pick").reset_index(drop=True)

def derive_roster_counts(draft_log: pd.DataFrame, players: pd.DataFrame, positions: List[str]) -> pd.DataFrame:
	draft_log = draft_log.copy()
	players = players.copy()
	draft_log["player_id"] = draft_log["player_id"].astype(str)
	players["player_id"] = players["player_id"].astype(str)
	pos_col = "pos" if "pos" in players.columns else "Pos"
	picks = draft_log[draft_log["player_id"].notna() & (draft_log["player_id"] != "None")].copy()
	if picks.empty:
		all_teams = pd.DataFrame({"team_slot": list(range(1, 1 + int(draft_log["team_slot"].max())))})
		for p in positions:
			all_teams[f"count_{p}"] = 0
		return all_teams
	picks_with_pos = picks.merge(players[["player_id", pos_col]], on="player_id", how="left")
	counts = picks_with_pos.groupby(["team_slot", pos_col]).size().unstack(fill_value=0)
	for p in positions:
		if p not in counts.columns:
			counts[p] = 0
	counts = counts[positions].reset_index()
	counts.columns = ["team_slot"] + [f"count_{p}" for p in positions]
	return counts

def compute_needs_and_propensities(
	roster_counts: pd.DataFrame,
	settings: Dict,
	market_pressure: Optional[Dict[int, Dict[str, float]]] = None,
	tier_pressure: Optional[Dict[int, Dict[str, float]]] = None,
	*,
	players_df: Optional[pd.DataFrame] = None,
	draft_log_df: Optional[pd.DataFrame] = None,
	use_internal_market_pressure: bool = False,
	market_top_k: int = 24,
	use_internal_tier_pressure: bool = True,
	tier_bonus_if_last: float = 0.30,
) -> pd.DataFrame:
	def _remaining_players(_players: pd.DataFrame, _draft: pd.DataFrame) -> pd.DataFrame:
		picked = set(_draft["player_id"].dropna().astype(str).tolist()) if ("player_id" in _draft.columns) else set()
		return _players[~_players["player_id"].astype(str).isin(picked)].copy()
	def _compute_market_pressure_from_adp(_players: pd.DataFrame, _draft: pd.DataFrame, _positions: List[str], top_k: int) -> Dict[int, Dict[str, float]]:
		remain = _remaining_players(_players, _draft)
		pos_col = "Pos" if "Pos" in remain.columns else "pos"
		if "ADP" in remain.columns: rank_col = "ADP"
		elif "ECR Rank" in remain.columns: rank_col = "ECR Rank"
		elif "adp_overall" in remain.columns: rank_col = "adp_overall"
		else:
			shares = {p: 0.0 for p in _positions}
			return {team: shares for team in range(1, settings["num_teams"] + 1)}
		remain[rank_col] = pd.to_numeric(remain[rank_col], errors="coerce")
		remain = remain.dropna(subset=[rank_col]).sort_values(rank_col)
		top = remain.head(top_k)
		counts = top.groupby(pos_col).size().reindex(_positions).fillna(0).astype(float)
		total = counts.sum()
		shares = {p: float(counts.loc[p] / total) if total > 0 else 0.0 for p in _positions}
		return {team: shares for team in range(1, settings["num_teams"] + 1)}
	def _compute_tier_pressure_simple(_players: pd.DataFrame, _draft: pd.DataFrame, _positions: List[str], bonus_if_last: float) -> Dict[int, Dict[str, float]]:
		remain = _remaining_players(_players, _draft)
		pos_col = "Pos" if "Pos" in remain.columns else "pos"
		tier_col = "Tier" if "Tier" in remain.columns else ("tier" if "tier" in remain.columns else None)
		if tier_col is None:
			zeros = {p: 0.0 for p in _positions}
			return {team: zeros for team in range(1, settings["num_teams"] + 1)}
		pressure = {}
		for p in _positions:
			pool = remain[remain[pos_col] == p]
			if pool.empty:
				pressure[p] = 0.0
				continue
			best_tier = pool[tier_col].min()
			count_best = int((pool[tier_col] == best_tier).sum())
			if count_best <= 2:
				scale = 1.0 if count_best <= 1 else 0.6
				pressure[p] = float(bonus_if_last * scale)
			else:
				pressure[p] = 0.0
		return {team: pressure for team in range(1, settings["num_teams"] + 1)}
	positions = settings.get("positions", ["QB", "RB", "WR", "TE", "K", "DEF"])
	all_teams = pd.DataFrame({"team_slot": list(range(1, settings["num_teams"] + 1))})
	rc = all_teams.merge(roster_counts, on="team_slot", how="left").fillna(0)
	starters_req = settings["starters_req"]
	roster_max = settings["roster_max"]
	weights = settings.get("needs_weights", {"alpha_starter": 1.0, "beta_bench": 0.5, "gamma_tier": 0.5, "delta_market": 0.25})
	sf = rc[["team_slot"]].copy()
	for col in rc.columns:
		if col.startswith("count_"):
			p = col.replace("count_", "")
			need = int(starters_req.get(p, 0))
			sf[f"starters_filled_{p}"] = np.minimum(rc[col].astype(int), need)
	needs_df = rc[["team_slot"]].copy()
	for p in positions:
		count_col = f"count_{p}"
		count_vals = rc[count_col].astype(int) if count_col in rc.columns else 0
		starter_need = np.maximum(0, starters_req.get(p, 0) - (sf.get(f"starters_filled_{p}", pd.Series(0)).fillna(0)))
		bench_room = np.maximum(0, roster_max.get(p, 0) - count_vals)
		needs_df[f"need_starter_{p}"] = starter_need
		needs_df[f"bench_room_{p}"] = bench_room
	bench_cols = [f"bench_room_{p}" for p in positions]
	bench_max = needs_df[bench_cols].max(axis=1).replace(0, 1)
	for p in positions:
		needs_df[f"bench_room_norm_{p}"] = needs_df[f"bench_room_{p}"] / bench_max
	if use_internal_market_pressure and (players_df is not None) and (draft_log_df is not None):
		market_pressure = _compute_market_pressure_from_adp(players_df, draft_log_df, positions, market_top_k)
	else:
		market_pressure = market_pressure or {}
	if use_internal_tier_pressure and (players_df is not None) and (draft_log_df is not None):
		tier_pressure = _compute_tier_pressure_simple(players_df, draft_log_df, positions, tier_bonus_if_last)
	else:
		tier_pressure = tier_pressure or {}
	prop_df = needs_df[["team_slot"]].copy()
	alpha = float(weights.get("alpha_starter", 1.0))
	beta = float(weights.get("beta_bench", 0.5))
	gamma = float(weights.get("gamma_tier", 0.5))
	delta = float(weights.get("delta_market", 0.25))
	for p in positions:
		starter_ind = (needs_df[f"need_starter_{p}"] > 0).astype(float)
		tier_vec = np.array([tier_pressure.get(team, {}).get(p, 0.0) for team in needs_df["team_slot"]])
		market_vec = np.array([market_pressure.get(team, {}).get(p, 0.0) for team in needs_df["team_slot"]])
		score = alpha * starter_ind + beta * needs_df[f"bench_room_norm_{p}"].astype(float) + gamma * tier_vec + delta * market_vec
		prop_df[f"score_{p}"] = score
	score_sum = np.sum([prop_df[f"score_{p}"] for p in positions], axis=0)
	score_sum = np.where(score_sum == 0, 1.0, score_sum)
	for p in positions:
		prop_df[f"propensity_{p}"] = prop_df[f"score_{p}"] / score_sum
	for p in positions:
		prop_df[f"need_starter_{p}"] = needs_df[f"need_starter_{p}"]
		prop_df[f"bench_room_{p}"] = needs_df[f"bench_room_{p}"]
	return prop_df

def get_team_pick_slots(team_slot: int, num_teams: int, num_rounds: int) -> List[int]:
	pick_slots = []
	for round_num in range(1, num_rounds + 1):
		if round_num % 2 == 1:
			pick_num = (round_num - 1) * num_teams + team_slot
		else:
			pick_num = (round_num - 1) * num_teams + (num_teams - team_slot + 1)
		pick_slots.append(pick_num)
	return pick_slots

def draft_player(draft_log_df: pd.DataFrame, player_id: str) -> pd.DataFrame:
	df = draft_log_df.copy()
	next_rows = df[df["player_id"].isna()].sort_values("overall_pick")
	if next_rows.empty:
		return df
	next_pick = int(next_rows.iloc[0]["overall_pick"])
	df.loc[df["overall_pick"] == next_pick, "player_id"] = str(player_id)
	return df

def compute_availability(
	players_df: pd.DataFrame,
	draft_log_df: pd.DataFrame,
	settings: Dict,
	my_team_slot: int,
	top_n: int = 30,
) -> pd.DataFrame:
	positions = settings.get("positions", ["QB", "RB", "WR", "TE", "K", "DEF"])
	players = normalize_players(players_df)
	drafted_players = draft_log_df[draft_log_df["player_id"].notna()]["player_id"].astype(str).tolist()
	players["player_id"] = players["player_id"].astype(str)
	remaining_players = players[~players["player_id"].isin(drafted_players)].copy()
	if remaining_players.empty:
		return pd.DataFrame(columns=["player_id","Name","Pos","Tier","ADP","Proj_Points","P_survive_to_next_pick","ADP_Value","Proj_Value","Notes"])
	adp_col = "adp_overall" if "adp_overall" in remaining_players.columns else ("ADP" if "ADP" in remaining_players.columns else None)
	proj_col = "proj_points" if "proj_points" in remaining_players.columns else ("Point Proj" if "Point Proj" in remaining_players.columns else None)
	if adp_col:
		remaining_players = remaining_players.sort_values(adp_col).head(top_n)
	current_pick = int(draft_log_df[draft_log_df["player_id"].isna()]["overall_pick"].min())
	my_pick_slots = get_team_pick_slots(my_team_slot, settings["num_teams"], settings["num_rounds"])
	my_next_pick = None
	for pick_num in sorted(my_pick_slots):
		row = draft_log_df[draft_log_df["overall_pick"] == pick_num]
		if not row.empty and pd.isna(row.iloc[0]["player_id"]):
			my_next_pick = pick_num
			break
	if my_next_pick is None:
		return pd.DataFrame(columns=["player_id","Name","Pos","Tier","ADP","Proj_Points","P_survive_to_next_pick","ADP_Value","Proj_Value","Notes"])
	roster_counts = derive_roster_counts(draft_log_df, players, positions)
	prop = compute_needs_and_propensities(
		roster_counts=roster_counts,
		settings=settings,
		players_df=players,
		draft_log_df=draft_log_df,
		use_internal_market_pressure=True,
		market_top_k=24,
		use_internal_tier_pressure=True,
		tier_bonus_if_last=0.30,
	)
	results: List[Dict] = []
	for _, player in remaining_players.iterrows():
		survival_prob = 1.0
		pos_col = "pos" if "pos" in player else "Pos"
		tier_col = "tier" if "tier" in player else "Tier"
		name_col = "name" if "name" in player else "Name"
		for pick_num in range(current_pick, my_next_pick):
			round_num = (pick_num - 1) // settings["num_teams"] + 1
			if round_num % 2 == 1:
				team_picking = pick_num - (round_num - 1) * settings["num_teams"]
			else:
				team_picking = settings["num_teams"] - (pick_num - (round_num - 1) * settings["num_teams"]) + 1
			team_props = prop[prop["team_slot"] == team_picking].iloc[0]
			if adp_col:
				sorted_players = remaining_players.sort_values(adp_col)
				player_rank = sorted_players.index.get_loc(player.name) + 1
			else:
				player_rank = remaining_players.index.get_loc(player.name) + 1
			base_weight = 1.0 / (player_rank ** 0.5)
			pos_propensity = team_props[f"propensity_{player[pos_col]}"]
			same_tier = remaining_players[remaining_players[tier_col] == player[tier_col]]
			tier_bonus = 1.3 if len(same_tier) <= 2 else 1.0
			pick_prob = base_weight * pos_propensity * tier_bonus
			total_weight = 0.0
			for _, other in remaining_players.iterrows():
				if adp_col:
					other_sorted = remaining_players.sort_values(adp_col)
					other_rank = other_sorted.index.get_loc(other.name) + 1
				else:
					other_rank = remaining_players.index.get_loc(other.name) + 1
				other_base = 1.0 / (other_rank ** 0.5)
				other_pos = other[pos_col]
				other_pos_prop = team_props[f"propensity_{other_pos}"]
				other_same_tier = remaining_players[remaining_players[tier_col] == other[tier_col]]
				other_tier_bonus = 1.3 if len(other_same_tier) <= 2 else 1.0
				total_weight += other_base * other_pos_prop * other_tier_bonus
			if total_weight > 0:
				pick_prob = pick_prob / total_weight
			survival_prob *= (1 - pick_prob)
		adp_value = player[adp_col] if adp_col else None
		proj_value = player[proj_col] if proj_col else None
		pick_probability = 1.0 - survival_prob
		adp_value_calc = (pick_probability * adp_value) if adp_value is not None else None
		proj_value_calc = (pick_probability * proj_value) if proj_value is not None else None
		results.append({
			"player_id": player["player_id"],
			"Name": player[name_col],
			"Pos": player[pos_col],
			"Tier": player[tier_col],
			"ADP": adp_value,
			"Proj_Points": proj_value,
			"P_survive_to_next_pick": survival_prob,
			"ADP_Value": adp_value_calc,
			"Proj_Value": proj_value_calc,
			"Notes": "",
		})
	return pd.DataFrame(results).sort_values("P_survive_to_next_pick", ascending=True)
