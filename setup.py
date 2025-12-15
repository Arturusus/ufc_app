# setup.py
from pathlib import Path
import sys

PROJECT = Path(".").resolve()

REQUIRED_ROOT_FILES = [
    "fighters.csv",
    "Fights.csv",
    "ufc_final_improved_model.h5",
    "ufc_scaler.pkl",
]

REQ_TXT = """\
fastapi
uvicorn
tensorflow
scikit-learn
joblib
numpy
pandas
"""

BACKEND_MAIN = r'''\
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib
import tensorflow as tf

from .ufc_infer import build_store, list_fights_between, make_features, get_card

app = FastAPI()

# ---- загрузка данных и модели ----
fighters_df = pd.read_csv("backend/fighters.csv")
fights_df = pd.read_csv("backend/Fights.csv")
store = build_store(fighters_df, fights_df)

model = tf.keras.models.load_model("backend/ufc_final_improved_model.h5")
scaler = joblib.load("backend/ufc_scaler.pkl")

class PredictBody(BaseModel):
    fighter1: str
    fighter2: str

@app.get("/api/fighters")
def get_fighters():
    return {"fighters": store["fighter_names"]}

@app.get("/api/fight/list")
def fight_list(f1: str, f2: str):
    return list_fights_between(store, f1, f2)

@app.post("/api/fight/predict")
def predict(body: PredictBody):
    if body.fighter1 == body.fighter2:
        raise HTTPException(status_code=400, detail="Нужно выбрать двух разных бойцов")

    try:
        x = make_features(store, body.fighter1, body.fighter2)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    x_scaled = scaler.transform([x])
    proba_f1 = float(model.predict(x_scaled, verbose=0)[0][0])
    pred = "Fighter_1" if proba_f1 > 0.5 else "Fighter_2"

    return {
        "predicted_winner": pred,
        "proba_fighter1_win": proba_f1,
        "fighter1": get_card(store, body.fighter1),
        "fighter2": get_card(store, body.fighter2),
    }

# UI (статика)
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
'''

BACKEND_INFER = r'''\
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
'''

# ВАЖНО: index.html берем из твоего "canvas" (то, что у тебя сейчас открыто)
UI_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>UFC Fight Predictor</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: Arial, sans-serif;
      background: #101018;
      color: #f5f5f5;
    }

    .container {
      max-width: 900px;
      margin: 20px auto;
      padding: 20px;
      background: #1b1b26;
      border-radius: 8px;
      box-shadow: 0 0 10px rgba(0,0,0,0.6);
    }

    h1 {
      text-align: center;
      margin-bottom: 20px;
      font-size: 2em;
    }

    .block {
      margin-bottom: 24px;
      padding: 12px;
      border-radius: 6px;
      background: #232331;
    }

    .block h2 {
      margin-bottom: 10px;
      font-size: 1.2em;
    }

    label {
      display: block;
      margin-top: 8px;
      font-weight: bold;
      font-size: 0.95em;
    }

    input[list], select {
      width: 100%;
      max-width: 400px;
      padding: 6px 8px;
      margin-top: 4px;
      margin-bottom: 8px;
      border: 1px solid #444;
      border-radius: 4px;
      background: #2b2b3a;
      color: #f5f5f5;
      font-size: 0.95em;
    }

    input[list]:focus, select:focus {
      outline: none;
      border-color: #e63946;
    }

    button {
      padding: 8px 16px;
      margin: 4px 4px 4px 0;
      border: none;
      border-radius: 4px;
      background: #e63946;
      color: #fff;
      cursor: pointer;
      font-size: 0.95em;
      transition: background 0.2s;
    }

    button:disabled {
      background: #555;
      cursor: not-allowed;
    }

    button:hover:not(:disabled) {
      background: #ff4b5c;
    }

    .result {
      margin-top: 12px;
      padding: 10px;
      min-height: 20px;
      background: #2b2b3a;
      border-left: 3px solid #e63946;
      border-radius: 3px;
    }

    .hint {
      font-size: 0.85em;
      color: #aaa;
      margin-top: 8px;
    }

    .buttons-row {
      margin-top: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    select {
      min-width: 280px;
    }

    .cards {
      display: flex;
      flex-wrap: wrap;
      margin-top: 12px;
      gap: 12px;
    }

    .card {
      background: #2b2b3a;
      padding: 12px;
      border-radius: 4px;
      flex: 1 1 300px;
      border: 1px solid #444;
    }

    .card h3 {
      margin-top: 0;
      margin-bottom: 8px;
      color: #e63946;
    }

    .card p {
      margin: 4px 0;
      font-size: 0.9em;
    }

    .loading {
      color: #aaa;
      font-style: italic;
    }

    .mode-label {
      display: inline-block;
      margin-top: 12px;
      padding: 6px 10px;
      background: #444;
      border-radius: 3px;
      font-size: 0.85em;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🥊 UFC Fight Predictor</h1>

    <div class="block">
      <h2>Выбор бойцов</h2>
      <label>Боец 1:</label>
      <input list="fighters" id="fighter1" placeholder="Начните печатать имя" />
      <label>Боец 2:</label>
      <input list="fighters" id="fighter2" placeholder="Начните печатать имя" />
      <datalist id="fighters"></datalist>
      <button id="btnLoadFights">Загрузить бои пары</button>
    </div>

    <div class="block">
      <h2>Режим 1: Угадай исход (по факту)</h2>
      <p class="hint">Доступен только если бойцы уже дрались в датасете.</p>
      <label>Выберите конкретный бой:</label>
      <select id="fightSelect" disabled></select>
      <div class="buttons-row">
        <button id="btnPickF1" disabled>Победит Fighter 1</button>
        <button id="btnPickF2" disabled>Победит Fighter 2</button>
      </div>
      <div id="mode1Result" class="result"></div>
    </div>

    <div class="block">
      <h2>Режим 2: Прогноз нейросети</h2>
      <p class="hint">Доступен всегда. Модель предсказывает победителя по характеристикам.</p>
      <button id="btnPredict">Предсказать</button>
      <div id="mode2Result" class="result"></div>
      <div id="fightersInfo" class="cards"></div>
    </div>

    <div class="block" style="background: #1a1a22; border-top: 1px solid #444; margin-top: 30px; text-align: center; font-size: 0.85em; color: #888;">
      <p>UFC Fight Predictor v1.0 | Backend: FastAPI + Keras | Data: Fighters.csv, Fights.csv</p>
    </div>
  </div>

  <script>
    // === DOM Elements ===
    const fighter1Input = document.getElementById("fighter1");
    const fighter2Input = document.getElementById("fighter2");
    const fightersDatalist = document.getElementById("fighters");

    const btnLoadFights = document.getElementById("btnLoadFights");
    const fightSelect = document.getElementById("fightSelect");
    const btnPickF1 = document.getElementById("btnPickF1");
    const btnPickF2 = document.getElementById("btnPickF2");
    const mode1Result = document.getElementById("mode1Result");

    const btnPredict = document.getElementById("btnPredict");
    const mode2Result = document.getElementById("mode2Result");
    const fightersInfo = document.getElementById("fightersInfo");

    // === State ===
    let fightsCache = [];
    let selectedFight = null;
    let allFighters = [];

    // === Helpers ===
    function setMode1Enabled(enabled) {
      fightSelect.disabled = !enabled;
      btnPickF1.disabled = true;
      btnPickF2.disabled = true;
      selectedFight = null;
      mode1Result.textContent = enabled
        ? "Выберите конкретный бой из списка, затем сделайте прогноз."
        : "❌ Этот режим недоступен: по этой паре боев в датасете нет.";
    }

    function fillFightSelect(fights) {
      fightSelect.innerHTML = "";
      fightSelect.appendChild(new Option("— Выберите бой —", ""));
      for (const f of fights) {
        const label = `${f.event_id || "Event"} | R${f.round} | ${f.fight_time} | ${f.method}`;
        fightSelect.appendChild(new Option(label, String(f.fight_id)));
      }
    }

    // === API calls ===
    async function loadFighters() {
      try {
        const res = await fetch("/api/fighters");
        const data = await res.json();
        allFighters = data.fighters || [];
        fightersDatalist.innerHTML = "";
        for (const name of allFighters) {
          const opt = document.createElement("option");
          opt.value = name;
          fightersDatalist.appendChild(opt);
        }
      } catch (e) {
        console.error("Ошибка загрузки бойцов:", e);
        mode1Result.textContent = "⚠️ Ошибка подключения к серверу.";
      }
    }

    async function loadFightList(f1, f2) {
      try {
        const url = `/api/fight/list?f1=${encodeURIComponent(f1)}&f2=${encodeURIComponent(f2)}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(res.status);
        const data = await res.json();

        if (!data.exists || !data.fights.length) {
          fightsCache = [];
          setMode1Enabled(false);
          return;
        }

        fightsCache = data.fights;
        fillFightSelect(fightsCache);
        setMode1Enabled(true);
      } catch (e) {
        console.error("Ошибка загрузки боев:", e);
        mode1Result.textContent = "⚠️ Ошибка загрузки боев. Проверьте имена бойцов.";
        setMode1Enabled(false);
      }
    }

    async function predictFight(f1, f2) {
      try {
        const res = await fetch("/api/fight/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ fighter1: f1, fighter2: f2 }),
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || res.status);
        }

        const data = await res.json();
        const winnerUi =
          data.predicted_winner === "Fighter_1" ? "Fighter 1" : "Fighter 2";
        const p1 = (data.proba_fighter1_win * 100).toFixed(1);

        mode2Result.innerHTML = `
          <strong>Предсказание модели:</strong> Победит <strong style="color: #e63946;">${winnerUi}</strong><br/>
          P(Fighter 1 win) = <strong>${p1}%</strong>
        `;

        fightersInfo.innerHTML = "";
        if (data.fighter1) {
          fightersInfo.appendChild(renderCard(data.fighter1, "Fighter 1"));
        }
        if (data.fighter2) {
          fightersInfo.appendChild(renderCard(data.fighter2, "Fighter 2"));
        }
      } catch (e) {
        console.error("Ошибка предсказания:", e);
        mode2Result.textContent = `⚠️ Ошибка: ${e.message}`;
      }
    }

    function renderCard(card, label) {
      const div = document.createElement("div");
      div.className = "card";
      div.innerHTML = `
        <h3>${label}: ${card.name}</h3>
        <p><strong>Рост:</strong> ${card.height_cm.toFixed(1)} см</p>
        <p><strong>Размах:</strong> ${card.reach_cm.toFixed(1)} см</p>
        <p><strong>Рекорд:</strong> ${card.wins}-${card.losses}-${card.draws}</p>
        <p><strong>Всего боев:</strong> ${card.total_fights}</p>
        <p><strong>Win%:</strong> ${(card.win_pct * 100).toFixed(1)}%</p>
        <p><strong>Пояс:</strong> ${card.belt ? "✓ Да" : "Нет"}</p>
      `;
      return div;
    }

    // === Event listeners ===
    btnLoadFights.addEventListener("click", async () => {
      const f1 = fighter1Input.value.trim();
      const f2 = fighter2Input.value.trim();

      if (!f1 || !f2) {
        mode1Result.textContent = "⚠️ Выберите двух бойцов из списка.";
        setMode1Enabled(false);
        return;
      }
      if (f1 === f2) {
        mode1Result.textContent = "⚠️ Нужно выбрать двух разных бойцов.";
        setMode1Enabled(false);
        return;
      }

      mode1Result.textContent = "🔄 Загружаем бои...";
      await loadFightList(f1, f2);
    });

    fightSelect.addEventListener("change", () => {
      const id = fightSelect.value;
      if (!id) {
        selectedFight = null;
        btnPickF1.disabled = true;
        btnPickF2.disabled = true;
        return;
      }
      selectedFight = fightsCache.find((x) => String(x.fight_id) === id) || null;
      const enabled = !!selectedFight;
      btnPickF1.disabled = !enabled;
      btnPickF2.disabled = !enabled;
      mode1Result.textContent = enabled
        ? "✓ Бой выбран. Сделайте прогноз: кто выиграет?"
        : "Выберите конкретный бой.";
    });

    btnPickF1.addEventListener("click", () => {
      if (!selectedFight) return;
      mode1Result.textContent =
        selectedFight.winner === "Fighter_1"
          ? "✅ Ты прав! (по датасету)"
          : "❌ Ты ошибся. (по датасету)";
    });

    btnPickF2.addEventListener("click", () => {
      if (!selectedFight) return;
      mode1Result.textContent =
        selectedFight.winner === "Fighter_2"
          ? "✅ Ты прав! (по датасету)"
          : "❌ Ты ошибся. (по датасету)";
    });

    btnPredict.addEventListener("click", async () => {
      const f1 = fighter1Input.value.trim();
      const f2 = fighter2Input.value.trim();

      if (!f1 || !f2) {
        mode2Result.textContent = "⚠️ Выберите двух бойцов.";
        return;
      }
      if (f1 === f2) {
        mode2Result.textContent = "⚠️ Выберите двух разных бойцов.";
        return;
      }

      mode2Result.innerHTML = '<span class="loading">🔄 Вычисляю прогноз...</span>';
      fightersInfo.innerHTML = "";

      await predictFight(f1, f2);
    });

    // === Init ===
    window.addEventListener("load", () => {
      loadFighters();
    });
  </script>
</body>
</html>
"""

def die(msg: str, code: int = 1):
    print("\n[setup.py] " + msg)
    sys.exit(code)

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def main():
    missing = [f for f in REQUIRED_ROOT_FILES if not (PROJECT / f).exists()]
    if missing:
        die("Не найдены файлы в текущей папке:\n- " + "\n- ".join(missing) +
            "\n\nПоложи рядом с setup.py эти файлы:\n"
            "fighters.csv, Fights.csv, ufc_final_improved_model.h5, ufc_scaler.pkl")

    # создать структуру
    (PROJECT / "backend").mkdir(exist_ok=True)
    (PROJECT / "ui").mkdir(exist_ok=True)

    # перенести "тяжелые" файлы в backend
    for fname in REQUIRED_ROOT_FILES:
        src = PROJECT / fname
        dst = PROJECT / "backend" / fname
        if not dst.exists():
            dst.write_bytes(src.read_bytes())

    # записать код
    write_file(PROJECT / "backend" / "__init__.py", "")
    write_file(PROJECT / "backend" / "main.py", BACKEND_MAIN)
    write_file(PROJECT / "backend" / "ufc_infer.py", BACKEND_INFER)
    write_file(PROJECT / "ui" / "index.html", UI_INDEX_HTML)
    write_file(PROJECT / "requirements.txt", REQ_TXT)

    print("\n✅ Готово.")
    print("Дальше команды:")
    print("1) pip install -r requirements.txt")
    print("2) uvicorn backend.main:app --reload")
    print("3) открыть http://127.0.0.1:8000/")

if __name__ == "__main__":
    main()
