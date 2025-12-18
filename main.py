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

# TODO Debug

# 1. Load Environment Variables and Constants
load_dotenv()
API_KEY = os.getenv("MARVEL_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

TOKEN_LIMIT = 1_000_000

# 2. Configuration & Setup
st.set_page_config(page_title="Marvel Rivals Analyst", page_icon="🕷️")
st.title("🕷️ Marvel Rivals Hero Analyst")

# --- PART A: API HElPERS ---

def get_player_data(player_id):
    """
    Fetches player profile and recent match summaries.
    """
    url = f"https://marvelrivalsapi.com/api/v1/player/{player_id}"
    headers = {'x-api-key': API_KEY}

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error fetching {player_id}: {e}")
        return None

def get_match_detail(match_uid):
    """
    Fetches detailed stats for a specific match.
    """
    url = f"https://marvelrivalsapi.com/api/v1/match/{match_uid}"
    headers = {'x-api-key': API_KEY}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error fetching match {match_uid}: {e}")
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
            "hit_rate": f"{round(hero.get('session_hit_rate', 0) * 100, 1)}%" # Note that this should be considered a percentage
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
    Tracks performance for EVERY hero played by the user in that match.
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
# List of heroes
TARGET_HEROES = [
    "Adam Warlock",
    "Angela",
    "Black Panther",
    "Black Widow",
    "Blade",
    "Bruce Banner",
    "Captain America",
    "Cloak & Dagger",
    "Daredevil",
    "Doctor Strange",
    "Emma Frost",
    "Gambit",
    "Groot",
    "Hawkeye",
    "Hela",
    "Human Torch",
    "Invisible Woman",
    "Iron Fist",
    "Iron Man",
    "Jeff The Land Shark",
    "Loki",
    "Luna Snow",
    "Magik",
    "Magneto",
    "Mantis",
    "Mister Fantastic",
    "Moon Knight",
    "Namor",
    "Peni Parker",
    "Phoenix",
    "Psylocke",
    "Rocket Raccoon",
    "Scarlet Witch",
    "Spider-Man",
    "Squirrel Girl",
    "Star-Lord",
    "Storm",
    "The Punisher",
    "The Thing",
    "Thor",
    "Ultron",
    "Venom",
    "Winter Soldier",
    "Wolverine"
]

def create_hero_id_map(hero_db):
    """
    Creates a reverse lookup dictionary: { 1053: 'Emma Frost', ... }
    """
    id_map = {}
    for hero_name, data in hero_db.items():
        if data and 'id' in data:
            # key = ID (int), value = Name (str)
            id_map[data['id']] = data['name']
    return id_map

def sanitize_hero_data(raw_data):
    """
    Strips out visual assets (URLs, icons) and huge lists (skins)
    to save LLM tokens while keeping gameplay-relevant info.
    """
    if not raw_data:
        return None

    # extract only what the LLM needs to know about gameplay/lore
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

    # Transformations
    for transformation in raw_data.get("transformations", []):
        clean_transformation = {
            "id": transformation.get("transformation_id"),
            "name": transformation.get("name"),
            "health": transformation.get("health"),
            "movement_speed": transformation.get("movement_speed")
        }
        clean_data["transformations"].append(clean_transformation)

    # Abilities
    for ability in raw_data.get("abilities", []):
        clean_ability = {
            "id": ability.get("ability_id"),
            "transformation_id": transformation.get("transformation_id"),
            "name": ability.get("name"),
            "type": ability.get("type"),
            "description": ability.get("description"),
            "additional_fields": ability.get("additional_fields")
        }
        clean_data["abilities"].append(clean_ability)

    return clean_data

@st.cache_data(ttl=3600, show_spinner="Fetching Hero Data...")
def initialize_hero_db():
    """
    Iterates through TARGET_HEROES, fetches data, sanitizes it,
    and returns a dict: { 'spider-man': { ...clean_data... } }
    """
    hero_db = {}
    headers = {'x-api-key': API_KEY}
    base_url = "https://marvelrivalsapi.com/api/v1/heroes/hero/"

    # Progress bar for UI Feedback
    progress_bar = st.progress(0, text="Initializing Hero Database...")

    for i, hero_name in enumerate(TARGET_HEROES):
        url = f"{base_url}{hero_name}"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                raw_data = response.json()
                hero_db[hero_name.lower()] = sanitize_hero_data(raw_data)
            else:
                print(f"Skipping hero {hero_name}: Status Code: {response.status_code}")
        except Exception as e:
            print(f"Skipping hero {hero_name}: {e}")

        # Update progress
        progress_bar.progress((i + 1) / len(TARGET_HEROES), text=f"Loaded {hero_name}...")
        time.sleep(0.1)  # Wait a bit

    progress_bar.empty()
    return hero_db


# --- PART C: DATA RETRIEVAL LOGIC ---

def calculate_gemini_cost(token_count, input_cost=0.3):
    """
    Calculates the estimated input cost.
    Defaults to input price for Gemini 2.5 Flash per 1M tokens.
    """
    return (token_count / 1_000_000) * input_cost

def get_analysis_context(user_query, player_input, hero_db):
    """
    Builds the context string.
    1. Always includes relevant Hero Data.
    2. If player_input (which will most likely be the player's name) is present, fetches recent matches to find patterns.

    Auto-trims matches if context > 1M tokens, and calculates cost.
    """
    # 1. Initialize token counter
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        return ""

    # 2. Build Static Context (Hero Info + Query)
    #    This shouldn't be trimmed.
    static_context = " --- HERO DATABASE ---\n"
    query_lower = user_query.lower()
    for h_name, h_data in hero_db.items():
        static_context += f"Info for {h_name}: {h_data}\n"

    # 3. Add player context (Coach Layer)
    match_summaries = []
    player_profile_str = ""

    if player_input:
        with st.status("Fetching Coach Data...", expanded=True) as status:
            status.write("Fetching Player Profile...")
            player_data = get_player_data(player_input)

            if player_data:
                # Get the UID
                uid = player_data.get("uid")

                # Add high level stats
                player_profile_str += f"\n--- PLAYER PROFILE ({player_data.get('name')}) ---\n"
                stats = player_data.get('overall_stats', {})
                ranked = stats.get('ranked', {})
                player_profile_str += f"TOTAL MATCHES: {stats.get('total_matches')}\n"
                player_profile_str += f"TOTAL WINS: {stats.get('total_wins')}\n"
                player_profile_str += f"TOTAL RANKED MATCHES: {ranked.get('total_matches')}\n"
                player_profile_str += f"TOTAL RANKED WINS: {ranked.get('total_wins')}\n"
                player_profile_str += f"TOTAL RANKED KDA: {ranked.get('total_kills')}/{ranked.get('total_deaths')}/{ranked.get('total_assists')}\n"
                player_profile_str += f"TOTAL RANKED MVPs/SVPs: {ranked.get('total_mvp')}/{ranked.get('total_svp')}\n"

                # Get Match History
                status.write("Analyzing recent matches...")
                history = player_data.get("match_history", [])[:50]

                # Fetch details in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {executor.submit(get_match_detail, m["match_uid"]): m for m in history}
                    for future in concurrent.futures.as_completed(futures):
                        details = future.result()
                        minified = minify_match_data(details, uid, create_hero_id_map(hero_db))

                        if minified:
                            match_summaries.append(minified)

                status.update(label="Coach Data Ready!", state="complete", expanded=False)
            else:
                static_context += "\n(Could not fetch player data. API might be down or ID invalid.)\n"

    # 4. Trimming Loop
    # Combine Static + Profile + Matches and check tokens.
    # If > TOKEN_LIMIT, remove the oldest match

    final_context_str = ""

    # Create a progress bar for the trimming process
    trim_message = st.empty()

    while True:
        # Construct dynamic match string
        matches_str = f"\n--- RECENT MATCH PERFORMANCE (Last {len(match_summaries)} Matches) ---\n"
        matches_str += json.dumps(match_summaries, indent=2)

        # Combine everything
        final_context_str = static_context + player_profile_str + matches_str

        # Count tokens
        token_count = client.models.count_tokens(model="gemini-2.5-flash", contents=final_context_str).total_tokens

        if token_count <= TOKEN_LIMIT:
            # Within limits!
            cost = calculate_gemini_cost(token_count)

            # Output metrics to UI
            col1, col2, col3 = st.columns(3)
            col1.metric("Token Count", f"{token_count:,}", help="Total tokens sent to Gemini")
            col2.metric("Matches Included", len(match_summaries), help="Matches fit in context")
            col3.metric("Estimated Input Cost", f"${cost:.4f}", help="Based on Gemini 2.5 Flash Pricing")

            if len(match_summaries) == 0 and token_count > TOKEN_LIMIT:
                st.error("❌ Even with 0 matches, the context is too big! Reduce Hero DB size.")

            break
        else:
            # We are over the limit
            if not match_summaries:
                break  # Can't trime anymore

            # Remove the oldest match
            removed_match = match_summaries.pop()
            trim_message.warning(
                f"⚠️ Context exceeded 1M tokens ({token_count:,}). Removing oldest match ({removed_match.get('match_uid')})...")

    trim_message.empty()
    return final_context_str

# --- PART D: APP INTERFACE ---

# Sidebar for Player Setup
with st.sidebar:
    st.header("Player Setup")
    player_uid_input = st.text_input("Enter Player Name", value="FlipFlopper")

    if "hero_db" not in st.session_state:
        st.session_state.hero_db = initialize_hero_db()

    st.success("Hero Database Loaded")

    # DEBUG for hero_db
    with st.expander("Debug: Hero Database"):
        st.write(f"Total Heroes Loaded:** {len(st.session_state.hero_db)}")
        st.json(st.session_state.hero_db)

# Initialize Chat Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "I'm ready to coach! Ask me about specific matchups."
        }
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- PART E: CHAT INTERFACE ---
if prompt := st.chat_input("Ask your coach..."):
    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    with st.chat_message(prompt):
        st.markdown(prompt)

    # Build Context
    final_context = get_analysis_context(prompt, player_uid_input, st.session_state.hero_db)

    # Construct prompt
    template = """
    You are an expert eSports Coach for Marvel Rivals.
    
    GOAL: Answer the user's question using the provided data.
    
    DATA SOURCES:
    1. HERO DATABASE: General stats and ability info.
    2. RECENT MATCH PERFORMANCE: The user's actual last 5 games.
    
    INSTRUCTIONS:
    - If the user asks about a specific enemy (e.g., "How do I do against Dr. Strange?"), look at the 'enemy_team_composition' in the match history. 
    - Identify if they won or lost those specific games and what their K/D/A was.
    - Combine this with the Hero Database to give advice (e.g., "You struggled against Dr. Strange in your last match (Match ID ending in 123), dying 4 times. Dr. Strange is a Vanguard, so try to...")
    
    CONTEXT:
    {context}
    
    USER QUESTION:
    {question}
    """

    custom_prompt = PromptTemplate.from_template(template)
    chain = custom_prompt | llm | StrOutputParser()

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = chain.invoke(
                    {
                        "context": final_context,
                        "question": prompt
                    }
                )
                st.markdown(response)

                # Save to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

                # Debug Expander
                with st.expander("Debug Context"):
                    st.text(final_context)
            except Exception as e:
                st.error(f"An error occurred: {e}")