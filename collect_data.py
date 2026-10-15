import requests
import csv
from datetime import datetime, timezone
import numpy as np
import ast
from datetime import datetime, timezone

# ============================================================
# CONFIG
# ============================================================

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJTdWJqZWN0IjoiMzliNTliNjUtNmI1OC00MGFmLWE0MzEtNDdmYWYxMWVhMDVlIiwiU3RlYW1JZCI6IjE0OTE5NDczOSIsIkFQSVVzZXIiOiJ0cnVlIiwibmJmIjoxNzkxMDIyOTg3LCJleHAiOjE4MjI1NTg5ODcsImlhdCI6MTc5MTAyMjk4NywiaXNzIjoiaHR0cHM6Ly9hcGkuc3RyYXR6LmNvbSJ9.fpm7gafT4TANe1my9oCTvN-nIz7pDGEvmyGgaflXBE4"



URL = "https://api.stratz.com/graphql"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "User-Agent": "STRATZ_API",
    "Content-Type": "application/json",
}

STEAM_ID_64 = 76561198109460467  

STEAM_ID = STEAM_ID_64 - 76561197960265728


# ADD MORE STUFF


# ============================================================
# ACCESSING MATCH IDs
# ============================================================


all_matches = []
take = 100
skip = 0

while True:
    query_matches = f"""
    query {{
        player(steamAccountId: {STEAM_ID}) {{
            matches(request: {{ take: {take}, skip: {skip} }}) {{
                id
                startDateTime
            }}
        }}
    }}
    """

    response = requests.post(
        URL,
        headers=HEADERS,
        json={"query": query_matches}
    )
    response.raise_for_status()

    result = response.json()

    if "errors" in result:
        raise RuntimeError(result["errors"])

    matches = result["data"]["player"]["matches"]

    if not matches:
        break

    all_matches.extend(matches)
    skip += len(matches)

    print(f"{len(all_matches)} Matches gesammelt")

match_ids = [match["id"] for match in all_matches]

print("Insgesamt:", len(match_ids))
print("Neueste Match-ID:", match_ids[0])
print("Älteste Match-ID:", match_ids[-1])

match = all_matches[1499]

print("Match ID:", match["id"])
print("Datum:", datetime.fromtimestamp(
    match["startDateTime"], tz=timezone.utc
))

# !take first 1500 entries to have games of approx last 4 years and store them in array (1500xapprox 20 ) - train on 1200, validate on 100 and test on 200
# !preprocessing needs to be applied (CARE: NOT FOR EVERYTHING - e.g. not for hero id - also check other properties for their distribution bc outliers might kill the preprocessing)

# -> richtigkeit der daten gegen recent match crosschecken (einfach in dota gucken -> vor preprocessing natürlich)

# für preprocessen nur dei trainingsdaten nehmen


"""
Liste mit welche Position was ist ####

x[0]
x[1]
....


"""