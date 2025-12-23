# TODO Consider architecture change. Instead of just looking at the MAX_MATCH_HISTORY_LEN most recent matches...
#   - When asked about match history, convert the user's query into a valid request for the API
#     Example: "How well did I do in Season 3.5?" -> Only look at match history in season 3.5
#   - For vague requests about match history (e.g., How well did I do against Doctor Strange), default to retrieving
#     the MAX_MATCH_HISTORY_LEN most recent matches
# TODO Consider caching my MAX_MATCH_HISTORY_LEN most recent matches

import os
import requests
import json
import time
import concurrent.futures
import csv
import io
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 1. Load Environment Variables and Constants
load_dotenv()
API_KEY = os.getenv("MARVEL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuration
MODEL_NAME = "gpt-5-mini-2025-08-07"

TOKEN_LIMIT = 400000
MAX_MATCH_HISTORY_LEN = 50

# Seasons to check (in order of priority)
TARGET_SEASONS = [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1]

# 2. Configuration & Setup
st.set_page_config(page_title="Marvel Rivals Analyst", page_icon="🕷️", layout="wide")

# Initialize Session State Variables
if "analysis_active" not in st.session_state:
    st.session_state.analysis_active = False
if "llm_context" not in st.session_state:
    st.session_state.llm_context = ""
if "match_history_data" not in st.session_state:
    st.session_state.match_history_data = []
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "I'm ready to coach! Ask me about specific matchups or your recent performance."}
    ]


# --- PART A: API HELPERS (Unchanged) ---

def get_player_data(player_identifier):
    """Fetches player profile (V1) to resolve Name -> UID and get high-level stats."""
    url = f"https://marvelrivalsapi.com/api/v1/player/{player_identifier}"
    headers = {'x-api-key': API_KEY}
    try:
        response = requests.get(url, headers=headers)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"Error fetching profile {player_identifier}: {e}")
        return None


def fetch_v2_match_history(player_uid, max_matches=MAX_MATCH_HISTORY_LEN):
    """Iterates through seasons using the V2 endpoint to find the most recent matches."""
    all_history_items = []
    headers = {'x-api-key': API_KEY}

    status_text = st.empty()

    for season in TARGET_SEASONS:
        if len(all_history_items) >= max_matches:
            break

        status_text.text(f"Scanning Season {season} history...")
        url = f"https://marvelrivalsapi.com/api/v2/player/{player_uid}/match-history"
        params = {"season": season, "skip": 0, "limit": MAX_MATCH_HISTORY_LEN}

        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                matches = data.get("match_history", [])
                if matches:
                    all_history_items.extend(matches)
        except Exception as e:
            print(f"Error fetching season {season}: {e}")

    status_text.empty()

    # Sort by timestamp descending (newest first)
    all_history_items.sort(key=lambda x: x.get('match_time_stamp', 0), reverse=True)
    return all_history_items[:max_matches]


def get_match_detail(match_uid):
    """Fetches detailed stats for a specific match (V1)."""
    url = f"https://marvelrivalsapi.com/api/v1/match/{match_uid}"
    headers = {'x-api-key': API_KEY}
    try:
        response = requests.get(url, headers=headers)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"Error fetching match detail {match_uid}: {e}")
        return None


def process_player_stats_in_match(player_data, hero_id_map):
    """Helper to structure player stats."""
    heroes_played_stats = []
    for hero in player_data.get("player_heroes", []):
        hero_id = hero.get("hero_id")
        hero_name = hero_id_map.get(str(hero_id), f"Unknown Hero: {hero_id}")
        heroes_played_stats.append({
            "hero_id": hero_id,
            "hero_name": hero_name,
            "play_time_seconds": round(hero.get("play_time", 0), 1),
            "kills": hero.get("kills", 0),
            "deaths": hero.get("deaths", 0),
            "assists": hero.get("assists", 0),
            "hit_rate": f"{round(hero.get('session_hit_rate', 0) * 100, 1)}%"
        })

    return {
        "player_uid": player_data.get("player_uid"),
        "aggregated_stats": {
            "total_kills": player_data.get("kills", 0),
            "total_deaths": player_data.get("deaths", 0),
            "total_assists": player_data.get("assists", 0),
            "total_hero_damage": player_data.get("total_hero_damage", 0),
            "total_hero_heal": player_data.get("total_hero_heal", 0),
            "total_damage_taken": player_data.get("total_damage_taken", 0),
        },
        "heroes_played_stats": heroes_played_stats
    }


def minify_match_data(full_match_json, target_player_uid, hero_id_map, season=None):
    """Extracts only essential data for the LLM."""
    if not full_match_json or "match_details" not in full_match_json:
        return None

    md = full_match_json["match_details"]
    match_uid = md.get("match_uid")
    target_camp = -1
    user_data = None

    for p in md.get("match_players", []):
        if str(p.get("player_uid")) == str(target_player_uid):
            target_camp = p.get("camp")
            user_data = p
            break

    if not user_data:
        return None

    target_player_stats = process_player_stats_in_match(user_data, hero_id_map)
    enemies = []
    for p in md.get("match_players", []):
        if p.get("camp") != target_camp:
            enemies.append(process_player_stats_in_match(p, hero_id_map))

    return {
        "match_uid": match_uid,
        "match_season": season,
        "result": "WIN" if user_data.get("is_win") == 1 else "LOSE",
        "map_id": md.get("map_id"),
        "target_player": target_player_stats,
        "enemies": enemies
    }


def convert_history_to_csv(match_data):
    """Converts match list to CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    headers = [
        "Match ID", "Season", "Result", "Map ID", "Side", "Match Total Kills",
        "Match Total Deaths", "Match Total Assists", "Match Total Damage Done",
        "Match Total Healing Done", "Match Total Damage Taken", "Hero Name",
        "Hero Play Time (s)", "Hero Kills", "Hero Deaths", "Hero Assists"
    ]
    writer.writerow(headers)

    for m in match_data:
        # Common match metadata
        m_id = m.get("match_uid")
        season = m.get("match_season")
        result = m.get("result")
        map_id = m.get("map_id")

        # Process USER (Me)
        match_stats = m.get("target_player", {}).get("aggregated_stats", {})
        user_heroes = m.get("target_player", {}).get("heroes_played_stats", [])
        if not user_heroes:
            writer.writerow([
                m_id, season, result, map_id, "ME", 0, 0, 0, 0, 0, 0, "Unknown", 0, 0, 0, 0
            ])

        for h in user_heroes:
            writer.writerow([
                m_id, season, result, map_id, "ME",
                match_stats.get("total_kills", 0), match_stats.get("total_deaths", 0),
                match_stats.get("total_assists", 0), match_stats.get("total_hero_damage", 0),
                match_stats.get("total_hero_heal", 0), match_stats.get("total_damage_taken", 0),
                h.get("hero_name", "Unknown"), h.get("play_time_seconds", 0), h.get("kills", 0), h.get("deaths", 0),
                h.get("assists", 0)
            ])

        # Process ENEMIES
        # 'enemies' is a list of player objects, each containing a 'heroes_played_stats' list
        for enemy in m.get("enemies", []):
            match_stats = enemy.get("aggregated_stats", {})
            for h in enemy.get("heroes_played_stats", []):
                # Only log significant enemy playtime to save tokens
                if h.get("play_time_seconds", 0) > 30:
                    writer.writerow([
                        m_id, season, result, map_id, "ENEMY",
                        match_stats.get("total_kills", 0), match_stats.get("total_deaths", 0),
                        match_stats.get("total_assists", 0), match_stats.get("total_hero_damage", 0),
                        match_stats.get("total_hero_heal", 0), match_stats.get("total_damage_taken", 0),
                        h.get("hero_name", "Unknown"), h.get("play_time_seconds", 0), h.get("kills", 0),
                        h.get("deaths", 0), h.get("assists", 0)
                    ])
    return output.getvalue()


def calculate_overall_stats(match_history):
    """
    Condenses a large list of matches into a tiny text summary string.
    """
    if not match_history:
        return "No recent match history available."

    total_games = len(match_history)
    wins = sum(1 for m in match_history if m.get('result') == 'WIN')
    losses = total_games - wins
    win_rate = (wins / total_games) * 100

    # Count most played heroes in this dataset
    hero_counts = {}
    for m in match_history:
        # We look at the target_player's heroes
        heroes = m.get("target_player", {}).get("heroes_played_stats", [])
        for h in heroes:
            name = h.get("hero_name", "Unknown")
            hero_counts[name] = hero_counts.get(name, 0) + 1

    # Sort top 3 heroes
    top_heroes = sorted(hero_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    top_heroes_str = ", ".join([f"{h[0]} ({h[1]} games)" for h in top_heroes])

    return f"""
    OVERALL DATASET SUMMARY (Past {total_games} Matches):
    - Win Rate: {win_rate:.1f}% ({wins} Wins, {losses} Losses)
    - Top Heroes Played: {top_heroes_str}
    """


# --- PART B: HERO DATABASE (CACHED) ---
TARGET_HEROES = [
    "Adam Warlock", "Angela", "Black Panther", "Black Widow", "Blade", "Bruce Banner",
    "Captain America", "Cloak & Dagger", "Daredevil", "Doctor Strange", "Emma Frost",
    "Gambit", "Groot", "Hawkeye", "Hela", "Human Torch", "Invisible Woman", "Iron Fist",
    "Iron Man", "Jeff The Land Shark", "Loki", "Luna Snow", "Magik", "Magneto", "Mantis",
    "Mister Fantastic", "Moon Knight", "Namor", "Peni Parker", "Phoenix", "Psylocke",
    "Rocket Raccoon", "Scarlet Witch", "Spider-Man", "Squirrel Girl", "Star-Lord",
    "Storm", "The Punisher", "The Thing", "Thor", "Ultron", "Venom", "Winter Soldier",
    "Wolverine"
]


def create_hero_id_map(hero_db):
    id_map = {}
    for hero_name, data in hero_db.items():
        if data and 'id' in data:
            id_map[str(data['id'])] = data['name']
    return id_map


def sanitize_hero_data(raw_data):
    if not raw_data: return None
    clean_data = {
        "id": raw_data.get("id"), "name": raw_data.get("name"),
        "role": raw_data.get("role"), "attack_type": raw_data.get("attack_type"),
        "difficulty": raw_data.get("difficulty"), "bio": raw_data.get("bio"),
        "abilities": []
    }
    for ability in raw_data.get("abilities", []):
        clean_data["abilities"].append({
            "name": ability.get("name"),
            "description": ability.get("description"),
        })
    return clean_data


@st.cache_data(ttl=3600, show_spinner="Fetching Hero Data...")
def initialize_hero_db():
    hero_db = {}
    headers = {'x-api-key': API_KEY}
    base_url = "https://marvelrivalsapi.com/api/v1/heroes/hero/"

    # We load silently to avoid UI clutter on every refresh
    for hero_name in TARGET_HEROES:
        url = f"{base_url}{hero_name}"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                raw_data = response.json()
                hero_db[hero_name.lower()] = sanitize_hero_data(raw_data)
        except:
            pass
    return hero_db


# --- PART C: CORE LOGIC FOR BUILD CONTEXT ---

def build_coach_context(player_input, hero_db):
    # Basic API Key Validation
    if not OPENAI_API_KEY:
        return "", [], "Error: OPENAI_API_KEY is missing from environment variables."

    # 1. Fetch Player Data
    player_data = get_player_data(player_input)
    if not player_data:
        return "", [], f"Could not find player: {player_input}"

    uid = player_data.get("uid")
    player_name = player_data.get('name')

    # 2. Fetch History (V2)
    v2_history_list = fetch_v2_match_history(uid, max_matches=MAX_MATCH_HISTORY_LEN)

    if not v2_history_list:
        return "", [], "Player found, but no match history detected."

    # 3. Parallel Fetch Details (V1)
    status_bar = st.progress(0, text="Downloading match details...")
    full_match_history = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_match_detail, m["match_uid"]): m for m in v2_history_list}
        completed = 0

        for future in concurrent.futures.as_completed(futures):
            details = future.result()
            v2_data = futures[future]
            season = v2_data.get("match_season")
            minified = minify_match_data(details, uid, create_hero_id_map(hero_db), season=season)
            if minified:
                full_match_history.append(minified)

            completed += 1
            status_bar.progress(completed / len(v2_history_list),
                                text=f"Analyzed match {completed}/{len(v2_history_list)}")

    status_bar.empty()

    # 1. Generate Summary of ALL downloaded matches
    dataset_summary = calculate_overall_stats(full_match_history)

    # 2. Generate CSV String for the LLM
    csv_history_str = convert_history_to_csv(full_match_history)

    # 3. Build the String
    static_context = " --- HERO DATABASE ---\n"
    for h_name, h_data in hero_db.items():
        compact_hero = {k: v for k, v in h_data.items() if k in ['name', 'role', 'abilities']}
        static_context += f"{h_name}: {compact_hero}\n"

    final_context_str = f"""
    {static_context}

    --- PLAYER PROFILE: {player_name} ---
    {dataset_summary}

    --- DETAILED MATCH LOGS (CSV FORMAT) ---
    The following data is in CSV format. Columns include Match ID, Season, Result, Map, and specific Hero Stats.

    {csv_history_str}
    """

    return final_context_str, full_match_history, f"Successfully loaded {len(full_match_history)} matches! (Converted to CSV for AI analysis)"


# --- PART D: APP INTERFACE ---

st.title("🕷️ Marvel Rivals Hero Analyst")

# Initialize Hero DB once
if "hero_db" not in st.session_state:
    with st.spinner("Initializing Hero Database..."):
        st.session_state.hero_db = initialize_hero_db()

# --- STATE 1: SETUP SCREEN ---
if not st.session_state.analysis_active:
    st.markdown("### 1. Player Setup")
    st.info(f"Enter a player name to download their last {MAX_MATCH_HISTORY_LEN} matches and analyze performance.")

    col1, col2 = st.columns([3, 1])
    with col1:
        player_input = st.text_input("Player Name", value="FlipFlopper", placeholder="Enter username...")
    with col2:
        st.write("")  # Spacer
        st.write("")
        if st.button("Fetch Match Data", type="primary", use_container_width=True):
            if not player_input:
                st.error("Please enter a player name.")
            else:
                context_str, matches, msg = build_coach_context(player_input, st.session_state.hero_db)
                if matches:
                    st.session_state.llm_context = context_str
                    st.session_state.match_history_data = matches
                    st.session_state.analysis_active = True
                    st.rerun()  # Refresh to show chat interface
                else:
                    st.error(msg)

# --- STATE 2: DASHBOARD & CHAT ---
else:
    # Top Bar: Actions
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.success(f"Data Loaded! ({len(st.session_state.match_history_data)} matches cached)")
    with c2:
        # Download Button
        csv_data = convert_history_to_csv(st.session_state.match_history_data)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="marvel_rivals_history.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c3:
        if st.button("🔄 New Search", use_container_width=True):
            st.session_state.analysis_active = False
            st.session_state.match_history_data = []
            st.session_state.llm_context = ""
            st.rerun()

    st.divider()

    # Chat Interface
    st.subheader("🤖 AI Coach Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask your coach about your performance..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # OpenAI Logic
        llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY)

        template = """
        ### ROLE & OBJECTIVE
        You are an elite eSports Coach for 'Marvel Rivals'.
        Your GOAL is to analyze the user's gameplay data to provide high-level, actionable strategic advice.
        
        ### TERMINOLOGY STANDARD
        - **Vanguard:** Tank (Focus: Space creation, mitigation)
        - **Duelist:** DPS (Focus: Securing kills, pressure)
        - **Strategist:** Healer/Support (Focus: Utility, sustain)
        
        ### DATA CONTEXT
        1. **HERO DATABASE:** Stats, abilities, and synergies.
        2. **MATCH HISTORY (CSV):** - **Columns:** Match_ID, Season, Result, Map, Side, Hero, Stats...
           - **'Side' Column:** "ME" indicates the user's stats. "ENEMY" indicates an opponent's stats in that same Match_ID.
           - **Analysis Tip:** To see who the user played against in Match X, look for rows where Match_ID = X and Side = "ENEMY".
        
        ### ANALYSIS PROTOCOL
        1. **Matchups:** Look at matches where Side="ME" resulted in a "LOSE". Cross-reference the "ENEMY" heroes in that same Match_ID. 
           *Example:* "You lost 3 games as Spider-Man when the Enemy had a Venom."
        2. **Trend Analysis:** patterns in KDA or Map types.
        3. **Hero Correlation:** Compare user performance against specific enemy compositions.
        
        ### OUTPUT GUIDELINES
        - **Be Concise:** Use bullet points.
        - **Identify Counters:** Specifically mention which Enemy heroes are causing the user trouble based on the CSV data.
        - **Win Condition:** End with one specific drill or focus area.
        
        ### DATA CONTEXT
        {context}
        
        ### USER QUESTION
        {question}
        """

        custom_prompt = PromptTemplate.from_template(template)
        chain = custom_prompt | llm | StrOutputParser()

        with st.chat_message("assistant"):
            with st.spinner("Analyzing match data..."):
                try:
                    response = chain.invoke({
                        "context": st.session_state.llm_context,
                        "question": prompt
                    })
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"AI Error: {e}")