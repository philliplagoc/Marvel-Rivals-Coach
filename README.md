# Marvel Rivals AI Coach

An LLM-powered chatbot that uses raw multiplayer match data for personalized coaching insights.

## Overview

* Fetches real player and match data from the [Marvel Rivals API](https://marvelrivalsapi.com/)
* Aggregates and normalizes multi-season match statistics
* Converts structured gameplay data into LLM-efficient context
* Uses OpenAI + LangChain to deliver performance insights via chat, and Streamlit for the interactive UI
* Identifies losing matchups, hero counters, and improvement areas

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

Requires:
* `MARVEL_API_KEY`: Get a key [here.](https://marvelrivalsapi.com/)
* `OPENAI_API_KEY`: Use OpenAI [to get a key.](https://platform.openai.com/docs/quickstart)
