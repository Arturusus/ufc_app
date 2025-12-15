\
import numpy as np
import pandas as pd

def convert_height(height):
    if pd.isna(height):
        return np.nan
    try:
        if isinstance(height, str) and "'" in height:
            feet, inches = height.split("'")
            feet = float(feet)
            inches = float(inches.replace('"', '').strip())
            return feet * 30.48 + inches * 2.54
        return float(height)
    except:
        return np.nan

def build_fighter_chars(fighters_df: pd.DataFrame):
    fighters_processed = fighters_df.copy()

    fighters_processed["height_cm"] = fighters_processed["Ht."].apply(convert_height)
    fighters_processed["reach_cm"] = fighters_processed["Reach"] * 2.54

    fighters_processed["height_cm"] = fighters_processed["height_cm"].fillna(
        fighters_processed["height_cm"].median()
    )
    fighters_processed["reach_cm"] = fighters_processed["reach_cm"].fillna(
        fighters_processed["reach_cm"].median()
    )
    fighters_processed["W"] = fighters_processed["W"].fillna(0)
    fighters_processed["L"] = fighters_processed["L"].fillna(0)
    fighters_processed["D"] = fighters_processed["D"].fillna(0)

    fighter_chars = {}
    for _, row in fighters_processed.iterrows():
        name = row["Full Name"]
        total_fights = row["W"] + row["L"] + row["D"]
        win_pct = row["W"] / (total_fights + 1e-10)

        fighter_chars[name] = {
            "height": float(row["height_cm"]),
            "reach": float(row["reach_cm"]),
            "wins": int(row["W"]),
            "losses": int(row["L"]),
            "draws": int(row["D"]),
            "total_fights": int(total_fights),
            "win_pct": float(win_pct),
            "belt": 1 if row.get("Belt") else 0,
        }

    return fighter_chars

def build_store(fighters_df: pd.DataFrame, fights_df: pd.DataFrame):
    fighter_chars = build_fighter_chars(fighters_df)
    names = sorted(fighter_chars.keys())
    return {
        "fighter_names": names,
        "fighter_chars": fighter_chars,
        "fights_df": fights_df,
    }

def list_fights_between(store, f1: str, f2: str):
    fights_df: pd.DataFrame = store["fights_df"]

    if f1 == f2:
        return {"exists": False, "fights": []}

    a = fights_df[(fights_df["Fighter_1"] == f1) & (fights_df["Fighter_2"] == f2)].copy()
    a["winner_ui"] = a["Result_1"].map(lambda r: "Fighter_1" if r == "W" else "Fighter_2")

    b = fights_df[(fights_df["Fighter_1"] == f2) & (fights_df["Fighter_2"] == f1)].copy()
    b["winner_ui"] = b["Result_1"].map(lambda r: "Fighter_2" if r == "W" else "Fighter_1")

    both = pd.concat([a, b], ignore_index=False)
    both = both[both["Result_1"].isin(["W", "L"])].copy()

    fights_out = []
    for idx, r in both.iterrows():
        fights_out.append(
            {
                "fight_id": int(idx),
                "event_id": str(r.get("Event_Id", "")),
                "round": int(r.get("Round", 0)) if pd.notna(r.get("Round", None)) else 0,
                "fight_time": str(r.get("Fight_Time", "")),
                "method": str(r.get("Method", "")),
                "winner": r["winner_ui"],
            }
        )

    return {"exists": len(fights_out) > 0, "fights": fights_out}

def make_features(store, f1: str, f2: str):
    fighter_chars = store["fighter_chars"]

    if f1 not in fighter_chars or f2 not in fighter_chars:
        raise ValueError("Нет данных по одному из бойцов")

    f1c = fighter_chars[f1]
    f2c = fighter_chars[f2]

    features = [
        f1c["height"] - f2c["height"],
        f1c["reach"] - f2c["reach"],
        f1c["wins"] - f2c["wins"],
        f1c["losses"] - f2c["losses"],
        f1c["win_pct"] - f2c["win_pct"],
        f1c["total_fights"] - f2c["total_fights"],

        f1c["height"],
        f1c["reach"],
        f1c["wins"],
        f1c["win_pct"],
        f1c["total_fights"],
        f1c["belt"],

        f2c["height"],
        f2c["reach"],
        f2c["wins"],
        f2c["win_pct"],
        f2c["total_fights"],
        f2c["belt"],

        f1c["height"] / (f2c["height"] + 1e-10),
        f1c["reach"] / (f2c["reach"] + 1e-10),
        f1c["win_pct"] / (f2c["win_pct"] + 1e-10),
    ]

    x = np.array(features, dtype=float)
    x = np.where(np.isinf(x), 100.0, x)
    return x

def get_card(store, name: str):
    fc = store["fighter_chars"].get(name)
    if not fc:
        return None
    return {
        "name": name,
        "height_cm": fc["height"],
        "reach_cm": fc["reach"],
        "wins": fc["wins"],
        "losses": fc["losses"],
        "draws": fc["draws"],
        "total_fights": fc["total_fights"],
        "win_pct": fc["win_pct"],
        "belt": bool(fc["belt"]),
    }
