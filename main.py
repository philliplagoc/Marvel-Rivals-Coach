# TODO Consider architecture change. Instead of just looking at the 50 most recent matches...
#   - When asked about match history, convert the user's query into a valid request for the API
#     Example: "How well did I do in Season 3.5?" -> Only look at match history in season 3.5
#   - For vague requests about match history (e.g., How well did I do against Doctor Strange), default to retrieving
#     the 50 most recent matches

import os
import requests
import json
import time
import concurrent.futures
import streamlit as st
from google import genai
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# 1. Load Environment Variables and Constants
load_dotenv()
API_KEY = os.getenv("MARVEL_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

TOKEN_LIMIT = 1_000_000

# Seasons to check (in order of priority)
TARGET_SEASONS = [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1]

# 2. Configuration & Setup
st.set_page_config(page_title="Marvel Rivals Analyst", page_icon="🕷️")
st.title("🕷️ Marvel Rivals Hero Analyst")


# --- PART A: API HElPERS ---

def get_player_data(player_identifier):
    """
    Fetches player profile (V1) to resolve Name -> UID and get high-level stats.
    """
    # Note: This endpoint handles both Name and ID resolution
    url = f"https://marvelrivalsapi.com/api/v1/player/{player_identifier}"
    headers = {'x-api-key': API_KEY}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error fetching profile {player_identifier}: {e}")
        return None


def fetch_v2_match_history(player_uid, max_matches=50):
    """
    Iterates through seasons using the V2 endpoint to find the most recent matches.
    Prioritizes Season 5 down to 1.
    """
    all_history_items = []
    headers = {'x-api-key': API_KEY}

    # Progress bar specifically for history fetching
    history_status = st.empty()

    for season in TARGET_SEASONS:
        if len(all_history_items) >= max_matches:
            break

        history_status.text(f"Checking Season {season} history...")

        # We request 50 to try and fill the buffer in one call per season
        url = f"https://marvelrivalsapi.com/api/v2/player/{player_uid}/match-history"
        params = {
            "season": season,
            "skip": 0,
            "limit": 50
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                matches = data.get("match_history", [])

                if matches:
                    all_history_items.extend(matches)
            else:
                print(f"Season {season} fetch failed: {response.status_code}")

        except Exception as e:
            print(f"Error fetching season {season}: {e}")

    history_status.empty()

    # Sort by timestamp descending (newest first) just in case seasons were mixed
    all_history_items.sort(key=lambda x: x.get('match_time_stamp', 0), reverse=True)

    # Return only the top N
    return all_history_items[:max_matches]


def get_match_detail(match_uid):
    """
    Fetches detailed stats for a specific match (V1).
    Required to get Enemy Team Composition.
    """
    url = f"https://marvelrivalsapi.com/api/v1/match/{match_uid}"
    headers = {'x-api-key': API_KEY}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error fetching match detail {match_uid}: {e}")
        return None


def process_player_stats_in_match(player_data, hero_id_map):
    """
    Processes important stats for a specific player for a specific match.
    Helper function for minify function.
    """
    # Process player's heroes
    heroes_played_stats = []
    for hero in player_data.get("player_heroes", []):
        hero_id = hero.get("hero_id")
        hero_name = hero_id_map.get(hero_id, f"Unknown Hero: {hero_id}")

        # Append specific stats
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


def minify_match_data(full_match_json, target_player_uid, hero_id_map):
    """
    Extracts only what the LLM needs to know to act as a coach.
    """
    if not full_match_json or "match_details" not in full_match_json:
        return None

    md = full_match_json["match_details"]
    match_uid = md.get("match_uid")

    # Identify user and their team camp
    target_camp = -1
    user_data = None

    # 1. Find target player
    for p in md.get("match_players", []):
        if str(p.get("player_uid")) == str(target_player_uid):
            target_camp = p.get("camp")
            user_data = p
            break

    if not user_data:
        return None

    # 2. Process User's stats
    target_player_stats = process_player_stats_in_match(user_data, hero_id_map)

    # 3. Identify enemy team comp
    enemies = []
    for p in md.get("match_players", []):
        if p.get("camp") != target_camp:
            enemies.append(process_player_stats_in_match(p, hero_id_map))

    # 4. Return minified match data
    return {
        "match_uid": match_uid,
        "result": "WIN" if user_data.get("is_win") == 1 else "LOSE",
        "map_id": md.get("map_id"),
        "target_player": target_player_stats,
        "enemies": enemies
    }


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
            id_map[data['id']] = data['name']
    return id_map


def sanitize_hero_data(raw_data):
    if not raw_data:
        return None

    clean_data = {
        "id": raw_data.get("id"),
        "name": raw_data.get("name"),
        "role": raw_data.get("role"),
        "attack_type": raw_data.get("attack_type"),
        "difficulty": raw_data.get("difficulty"),
        "bio": raw_data.get("bio"),
        "team": raw_data.get("team"),
        "lore": raw_data.get("lore"),
        "transformations": [],
        "abilities": []
    }

    for transformation in raw_data.get("transformations", []):
        clean_transformation = {
            "id": transformation.get("transformation_id"),
            "name": transformation.get("name"),
            "health": transformation.get("health"),
            "movement_speed": transformation.get("movement_speed")
        }
        clean_data["transformations"].append(clean_transformation)

    for ability in raw_data.get("abilities", []):
        clean_ability = {
            "id": ability.get("ability_id"),
            "name": ability.get("name"),
            "type": ability.get("type"),
            "description": ability.get("description"),
            "additional_fields": ability.get("additional_fields"),
        }
        clean_data["abilities"].append(clean_ability)

    return clean_data


@st.cache_data(ttl=3600, show_spinner="Fetching Hero Data...")
def initialize_hero_db():
    hero_db = {}
    headers = {'x-api-key': API_KEY}
    base_url = "https://marvelrivalsapi.com/api/v1/heroes/hero/"
    progress_bar = st.progress(0, text="Initializing Hero Database...")

    for i, hero_name in enumerate(TARGET_HEROES):
        url = f"{base_url}{hero_name}"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                raw_data = response.json()
                hero_db[hero_name.lower()] = sanitize_hero_data(raw_data)
        except Exception as e:
            print(f"Skipping hero {hero_name}: {e}")

        progress_bar.progress((i + 1) / len(TARGET_HEROES), text=f"Loaded {hero_name}...")
        time.sleep(0.05)

    progress_bar.empty()
    return hero_db


# --- PART C: DATA RETRIEVAL LOGIC ---

def calculate_gemini_cost(token_count, input_cost=0.3):
    return (token_count / 1_000_000) * input_cost


def get_analysis_context(user_query, player_input, hero_db):
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        return ""

    # 2. Build Static Context (Hero Info)
    static_context = " --- HERO DATABASE ---\n"
    for h_name, h_data in hero_db.items():
        static_context += f"Info for {h_name}: {h_data}\n"

    # 3. Add player context (Coach Layer)
    match_summaries = []
    player_profile_str = ""

    if player_input:
        with st.status("Fetching Coach Data...", expanded=True) as status:
            # Step A: Get UID via V1 Player Endpoint (still needed to resolve Name -> UID)
            status.write("Fetching Player Profile...")
            player_data = get_player_data(player_input)

            if player_data:
                uid = player_data.get("uid")

                # Add high level stats
                player_profile_str += f"\n--- PLAYER PROFILE ({player_data.get('name')}) ---\n"
                stats = player_data.get('overall_stats', {})
                ranked = stats.get('ranked', {})
                player_profile_str += f"TOTAL MATCHES: {stats.get('total_matches')}\n"
                player_profile_str += f"TOTAL WINS: {stats.get('total_wins')}\n"
                player_profile_str += f"RANKED TOTAL MATCHES: {ranked.get('total_matches')}\n"
                player_profile_str += f"RANKED WINS: {ranked.get('total_wins')}\n"
                player_profile_str += f"RANKED KDA: {ranked.get('total_kills')}/{ranked.get('total_deaths')}\n"

                # Step B: Get History via V2 Endpoint (Iterating Seasons)
                status.write("Scanning Seasons for Match History (V2)...")
                v2_history_list = fetch_v2_match_history(uid, max_matches=50)
                status.write(f"Found {len(v2_history_list)} recent matches.")

                # Step C: Parallel Fetch of Details (V1 Match Endpoint)
                # We need this because V2 history doesn't include Enemy Team Composition
                status.write("Downloading detailed match logs...")

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Map future to match_uid
                    futures = {
                        executor.submit(get_match_detail, m["match_uid"]): m
                        for m in v2_history_list
                    }

                    for future in concurrent.futures.as_completed(futures):
                        details = future.result()
                        # Minify will extract enemies and result
                        minified = minify_match_data(details, uid, create_hero_id_map(hero_db))

                        if minified:
                            match_summaries.append(minified)

                status.update(label="Coach Data Ready!", state="complete", expanded=False)
            else:
                static_context += "\n(Could not fetch player data. API might be down or ID invalid.)\n"

    # 4. Trimming Loop
    final_context_str = ""
    trim_message = st.empty()

    while True:
        matches_str = f"\n--- RECENT MATCH PERFORMANCE (Last {len(match_summaries)} Matches) ---\n"
        matches_str += json.dumps(match_summaries, indent=2)
        final_context_str = static_context + player_profile_str + matches_str

        token_count = client.models.count_tokens(model="gemini-2.5-flash", contents=final_context_str).total_tokens

        if token_count <= TOKEN_LIMIT:
            cost = calculate_gemini_cost(token_count)
            col1, col2, col3 = st.columns(3)
            col1.metric("Token Count", f"{token_count:,}")
            col2.metric("Matches Included", len(match_summaries))
            col3.metric("Est. Cost", f"${cost:.4f}")
            break
        else:
            if not match_summaries:
                break
            match_summaries.pop()  # Remove oldest (last in list)

    trim_message.empty()
    return final_context_str


# --- PART D: APP INTERFACE ---

with st.sidebar:
    st.header("Player Setup")
    player_uid_input = st.text_input("Enter Player Name", value="FlipFlopper")
    if "hero_db" not in st.session_state:
        st.session_state.hero_db = initialize_hero_db()
    st.success(f"Hero Database Loaded ({len(st.session_state.hero_db)} heroes)")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I'm ready to coach! Ask me about specific matchups."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- PART E: CHAT INTERFACE ---
if prompt := st.chat_input("Ask your coach..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message(prompt):
        st.markdown(prompt)

    final_context = get_analysis_context(prompt, player_uid_input, st.session_state.hero_db)

    template = """
    You are an expert eSports Coach for Marvel Rivals.
    GOAL: Answer the user's question using the provided data.
    DATA SOURCES:
    1. HERO DATABASE: General stats and ability info.
    2. RECENT MATCH PERFORMANCE: The user's actual last 50 games.
    INSTRUCTIONS:
    - If the user asks about a specific enemy, look at the 'enemy_team_composition' in the match history. 
    - Identify if they won or lost those specific games.
    
    Here's some lingo you should know:
    - Vanguards can also be referred to as Tanks.
    - Duelists can also be referred to as DPS i.e. heroes specializing in dealing lots of damage.
    - Strategists can also be referred to Healers or Supports.

    CONTEXT:
    {context}

    USER QUESTION:
    {question}
    """

    custom_prompt = PromptTemplate.from_template(template)
    chain = custom_prompt | llm | StrOutputParser()

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = chain.invoke({"context": final_context, "question": prompt})
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")