import requests
import csv
from datetime import datetime, timezone
import numpy as np
import ast


def get_single_match_data(match_id=8977382530, time=0):

    API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJTdWJqZWN0IjoiMzliNTliNjUtNmI1OC00MGFmLWE0MzEtNDdmYWYxMWVhMDVlIiwiU3RlYW1JZCI6IjE0OTE5NDczOSIsIkFQSVVzZXIiOiJ0cnVlIiwibmJmIjoxNzkxMDIyOTg3LCJleHAiOjE4MjI1NTg5ODcsImlhdCI6MTc5MTAyMjk4NywiaXNzIjoiaHR0cHM6Ly9hcGkuc3RyYXR6LmNvbSJ9.fpm7gafT4TANe1my9oCTvN-nIz7pDGEvmyGgaflXBE4"



    URL = "https://api.stratz.com/graphql"

    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "User-Agent": "STRATZ_API",
        "Content-Type": "application/json",
    }

    # ADD MORE STUFF

    # ============================================================
    # ACCESSING GAMES STATS
    # ============================================================

    # matches for testing: 9023591474, 8977382530 (replace in every query)


    query_match = f"""
    query {{
        match(id: {match_id}) {{
            id
            actualRank
            gameMode
            durationSeconds
            didRadiantWin
            startDateTime
        }}
    }}
    """

    response_match = requests.post(URL, headers=HEADERS, json={"query": query_match})

    print("Status:", response_match.status_code, " \n ")
    #print(response_match.json(), " \n \n ")

    matchdata = response_match.json()

    matchid = np.array(matchdata['data']['match']['id'])
    actualRank = np.array(int(matchdata['data']['match']['actualRank']))
    gameMode = np.array(matchdata['data']['match']['gameMode'])
    durationSeconds = np.array(int(matchdata['data']['match']['durationSeconds'])) # use to see how far we can go for this match: 0, 5, 10, 5, ... (max until 40 id say)
    didRadiantWin = np.array(matchdata['data']['match']['didRadiantWin'])
    startDateTime = np.array(matchdata['data']['match']['startDateTime'])


    print(f"Match ID: {matchid}, \nActual Rank: {actualRank}, \nGame Mode: {gameMode}, \nDuration: {durationSeconds} seconds, \nDid Radiant Win: {didRadiantWin}, \nStart DateTime: {startDateTime} \n")



    # ============================================================
    # CHECKING IF MATCH IS SUITABLE
    # ============================================================

    if int(actualRank) > 75 or int(actualRank) < 21:
        print("Match is not suitable for analysis. Actual rank is outside the desired range.")
        exit() # später mit continue replacen falls ichs in schleife mache und mehrere matches checke oder halt nicht falls ich code von andereer datei aus aufrufe

    if gameMode != "ALL_PICK_RANKED": # überlegen ob ich bei mehr matches auch ranked dann voraussetzen möchte
        print("Match is not suitable for analysis. Game mode is not All Pick.")
        exit() # später mit continue replacen falls ichs in schleife mache und mehrere matches checke oder halt nicht falls ich code von andereer datei aus aufrufe

    # ... insert any additional code here to check for further desired characteristics of the match

    # ============================================================
    # ACCESSING PLAYER  STATS
    # ============================================================

    # ... currently fixed match ID - rewrite later


    # barrack is a bit code that encodes for six barracks and results in a number, direKills, radiantKills, radiantNetworthLeads, radiantExperienceLeads is a list where every entry corresponds to 1 minute timeframe (need to take sum for kills, probably not for exp and gold but maybe check again) - there is one extra entry in the timeline (prob for 00:00)
    # tower status werden auch wieder bits sein -> zu bits und dann one hot encoding also 11er vektor und den als input nutzen (jeder eintrag 0 oder 1 je nachdem ob tower am leben oder nicht)
    # predicted winrates kann ich später mal auslesen und mein Modell damit vgl
    # networth müssten bspw die genauen networth werte und nicht die änderung sein - im kopf behalten und bei jeder größe immer checken, was es ist
    query_playerstats = f"""
    query {{
    match(id: {match_id}) {{
        barracksStatusDire
        barracksStatusRadiant
        direKills
        radiantKills
        radiantNetworthLeads
        radiantExperienceLeads
        durationSeconds 
        towerStatusDire
        towerStatusRadiant 
        players {{
        isRadiant
        heroId                                                
        stats {{
            level
            lastHitsPerMinute
            networthPerMinute
            deniesPerMinute
            killEvents {{
            time
            }}
            deathEvents {{
            time
            }}
            assistEvents {{
            time
            }}
        }}
        }}
                        
    }}
    }}
    """
    # Role ergänzen falls das consistent ist

    response_playerstats = requests.post(URL, headers=HEADERS, json={"query": query_playerstats})
    playerdata = response_playerstats.json()
    print("Status:", response_playerstats.status_code, " \n ")
    #print(response_playerstats.json(), " \n \n ")
    playerdata = response_playerstats.json()

    barracksStatusDire = np.array([int(x) for x in str(format(playerdata['data']['match']['barracksStatusDire'], "06b"))]) # vllt dann eher ein jnp.array später
    barracksStatusRadiant = np.array([int(x) for x in str(format(playerdata['data']['match']['barracksStatusRadiant'], "06b"))])
    towerStatusDire = np.array([int(x) for x in str(format(playerdata['data']['match']['towerStatusDire'], "011b"))])
    towerStatusRadiant = np.array([int(x) for x in str(format(playerdata['data']['match']['towerStatusRadiant'], "011b"))])
    #towerDeaths = playerdata['data']['match']['towerDeaths']
    #towerStatus = playerdata['data']['match']['towerStatus']

    print("Barracks Status Dire:", barracksStatusDire, "\nBarracks Status Radiant:", barracksStatusRadiant, "\nTower Status Dire:", towerStatusDire, "\nTower Status Radiant:", towerStatusRadiant, "\n")

    timesteps = durationSeconds // 60 // 5 +1 # +1 bc we also wann do it at 0
    if timesteps > 9:
        timesteps = 9

    print(f"Timesteps: {timesteps} \n")


    direKills = np.array([np.sum(np.array(playerdata['data']['match']['direKills'], dtype=int)[:5*t + 1]) for t in range(timesteps)])
    radiantKills = np.array([np.sum(np.array(playerdata['data']['match']['radiantKills'], dtype=int)[:5*t + 1]) for t in range(timesteps)])
    radiantNetworthLeads = np.array(playerdata['data']['match']['radiantNetworthLeads'], dtype=int)[1::5]
    radiantExperienceLeads = np.array(playerdata['data']['match']['radiantExperienceLeads'], dtype=int)[1::5]

    print(f"Dire Kills: {direKills}, \nRadiant Kills: {radiantKills}, \nRadiant Networth Leads: {radiantNetworthLeads}, \nRadiant Experience Leads: {radiantExperienceLeads} \n")

    players = playerdata['data']['match']['players']  
    isRadiant = np.array([player['isRadiant'] for player in players]) # danach kann man sie dann ordnen  
    heroIds = np.array([player['heroId'] for player in players]) 
    stats = [player['stats'] for player in players]
    level = [np.array(player['stats']['level']) for player in players]
    levels = np.array([np.array([np.sum(playerlevel < t*5*60) for t in range(timesteps)]) for playerlevel in level])
    lastHitsPerMinute = np.array([player['stats']['lastHitsPerMinute'] for player in players])
    lastHits = np.array([np.array([np.sum(playerlhpm[:5*t]) for t in range(timesteps)]) for playerlhpm in lastHitsPerMinute])
    networthPerMinute = np.array([player['stats']['networthPerMinute'] for player in players])
    networths = np.array([[playernw[5*t] for t in range(timesteps)] for playernw in networthPerMinute])
    deniesPerMinute = np.array([player['stats']['deniesPerMinute'] for player in players])
    denies = np.array([np.array([np.sum(playerdpm[:5*t]) for t in range(timesteps)]) for playerdpm in deniesPerMinute])
    killEvents = [player['stats']['killEvents'] for player in players]
    killEvents = str(killEvents)
    killEvents = killEvents.replace("{'time': ", "").replace("}", "")
    killEvents = ast.literal_eval(killEvents)
    kills = np.array([np.array([np.sum(np.array(playerkill) < t*5*60) for t in range(timesteps)]) for playerkill in killEvents])
    deathEvents = [player['stats']['deathEvents'] for player in players]
    deathEvents = str(deathEvents)
    deathEvents = deathEvents.replace("{'time': ", "").replace("}", "")
    deathEvents = ast.literal_eval(deathEvents)
    deaths = np.array([np.array([np.sum(np.array(playerdeath) < t*5*60) for t in range(timesteps)]) for playerdeath in deathEvents])
    assistEvents = [player['stats']['assistEvents'] for player in players]
    assistEvents = str(assistEvents)
    assistEvents = assistEvents.replace("{'time': ", "").replace("}", "")
    assistEvents = ast.literal_eval(assistEvents)
    assists = np.array([np.array([np.sum(np.array(playerassist) < t*5*60) for t in range(timesteps)]) for playerassist in assistEvents])


    print(f"Is Radiant:\n {isRadiant}", "\n", f"Hero ID:\n {heroIds}", "\n", f"\nLevel:\n {levels}", "\n", f"\nLast Hits Per Minute:\n {lastHits}", f"\nNetworth Per Minute:\n {networths}", f"\nDenies Per Minute:\n {denies}", f"\nKill Events:\n {kills}", f"\nDeath Events:\n {deaths}", f"\nAssist Events:\n {assists}")

    


    # jetzt hier einen output erzeugen, der jeweils alles zeitunabhängige vom match enthhält, und zusätzlich alles zeitabhängige match und player jeweils für einen Zeitpunkt (also insgesamt t+1 mal)
    # diesen code dann von anderem aus aufrufen, wo ich dann matches durchlaufe -> dafür in methode umschreiben (am besten so, dass man in methode zeit übergibt und dann hier nur der output für die zeit erzeugt wird)

    if time > timesteps:
        print(f"Time {time} is greater than available timesteps {timesteps}. Exiting.")
        exit()
    else:
        print(f"Generating output for time {time}...")
        game_identification = np.array([matchid, actualRank, gameMode, durationSeconds, startDateTime])
        game_winner = np.array([didRadiantWin])
        game_stats = np.array([direKills[time], radiantKills[time], radiantNetworthLeads[time], radiantExperienceLeads[time]])
        heroIds_Radiant = heroIds[isRadiant]
        heroIds_Dire = heroIds[~isRadiant]
        levels_Radiant = levels[isRadiant, time]
        levels_Dire = levels[~isRadiant, time]
        lastHits_Radiant = lastHits[isRadiant, time]
        lastHits_Dire = lastHits[~isRadiant, time]
        denies_Radiant = denies[isRadiant, time]
        denies_Dire = denies[~isRadiant, time]
        networths_Radiant = networths[isRadiant, time]
        networths_Dire = networths[~isRadiant, time]
        kills_Radiant = kills[isRadiant, time]
        kills_Dire = kills[~isRadiant, time]
        deaths_Radiant = deaths[isRadiant, time]
        deaths_Dire = deaths[~isRadiant, time]
        assists_Radiant = assists[isRadiant, time]
        assists_Dire = assists[~isRadiant, time]

        player1_Radiant = np.array([heroIds_Radiant[0], levels_Radiant[0], lastHits_Radiant[0], denies_Radiant[0], networths_Radiant[0], kills_Radiant[0], deaths_Radiant[0], assists_Radiant[0]])
        player2_Radiant = np.array([heroIds_Radiant[1], levels_Radiant[1], lastHits_Radiant[1], denies_Radiant[1], networths_Radiant[1], kills_Radiant[1], deaths_Radiant[1], assists_Radiant[1]])
        player3_Radiant = np.array([heroIds_Radiant[2], levels_Radiant[2], lastHits_Radiant[2], denies_Radiant[2], networths_Radiant[2], kills_Radiant[2], deaths_Radiant[2], assists_Radiant[2]])
        player4_Radiant = np.array([heroIds_Radiant[3], levels_Radiant[3], lastHits_Radiant[3], denies_Radiant[3], networths_Radiant[3], kills_Radiant[3], deaths_Radiant[3], assists_Radiant[3]])
        player5_Radiant = np.array([heroIds_Radiant[4], levels_Radiant[4], lastHits_Radiant[4], denies_Radiant[4], networths_Radiant[4], kills_Radiant[4], deaths_Radiant[4], assists_Radiant[4]])

        player1_Dire = np.array([heroIds_Dire[0], levels_Dire[0], lastHits_Dire[0], denies_Dire[0], networths_Dire[0], kills_Dire[0], deaths_Dire[0], assists_Dire[0]])
        player2_Dire = np.array([heroIds_Dire[1], levels_Dire[1], lastHits_Dire[1], denies_Dire[1], networths_Dire[1], kills_Dire[1], deaths_Dire[1], assists_Dire[1]])
        player3_Dire = np.array([heroIds_Dire[2], levels_Dire[2], lastHits_Dire[2], denies_Dire[2], networths_Dire[2], kills_Dire[2], deaths_Dire[2], assists_Dire[2]])
        player4_Dire = np.array([heroIds_Dire[3], levels_Dire[3], lastHits_Dire[3], denies_Dire[3], networths_Dire[3], kills_Dire[3], deaths_Dire[3], assists_Dire[3]])
        player5_Dire = np.array([heroIds_Dire[4], levels_Dire[4], lastHits_Dire[4], denies_Dire[4], networths_Dire[4], kills_Dire[4], deaths_Dire[4], assists_Dire[4]])

        return game_identification, game_winner, game_stats, player1_Radiant, player2_Radiant, player3_Radiant, player4_Radiant, player5_Radiant, player1_Dire, player2_Dire, player3_Dire, player4_Dire, player5_Dire
        
        # PLAN: GLEICHES NETWORK AUF JEDEM PLAYER SEINE WERTE (DAVOR LERNBARES EMBEDDING AUF HERO ID JEWEILS) + NETWORK AUF CONCATENATED OUTPUTS DER RADIANT UND DIRE (GUCKEN, DASS DAS UNABHÄNGIG VON INPUT REIHENFOLGE DER SPIELER !!!) UND EIN NETWORK AUF MATCH SACHEN - UND DANN AUF DIE 3 CONCATENTED OUTPUT WIEDER EIN NETWORKT

        

        # lh und denies drin damit netzt sieht welche hero carry ist etc
        # didradiantwin brauch ich dann in game_stats noch dazu

        # tower und barack stats rauslassen, weil die sind nur ende des games - rausfinden ob die geparsed werden

        # bei netzwerken ein hero embedding trainieren (chatgpt fragen wie - weil one-hot denk zu viel)


if __name__ == "__main__":
    get_single_match_data(match_id=9051673110, time=5)  



# gucken, dass alle Minuten in etwa gleich viel Training Data habem

# MLPs jeweils auf einzelne Spieler anwenden, dann die outputs von radiant (irgendeine dim =! 1) zusammenfügen  und daruf mlp und fleiches mit dire und dann noch eins mit den allgemeinen stats und die outputs dieser 3 mlps wieder vereinen

# gucken, ob role immer s.d. 1 von jeder rolle (und nicht sowas wie roaming) und dann könnte man immer ein mlp für pos 1, eins für pos 2 und so weiter bei denen die anfangs auf spieler angewendetweren (sonst gleiches mlp auf alle andwenden, also einfach input in höhere dim transofrmieren (vmap netzwerk über die 5 input arrays, aber netzwerk für radiant und dire players kann ggf anderes sein, also 2 in total (+ die 3 die noch zusätzluch kommen)))
# kann man zur einfachheit anfangs aber auch weglassen aber auf keinen fall pro player ein anderers netzwerk dann



# in bits und dann onehotencoden


# for testing certain inputs: 
#query_testing = """
#query {
#  kill: __type(name: "MatchPlayerStatsKillEventType") {
#    fields {
#      name
#      type {
#        kind
#        name
#        ofType {
#          kind
#          name
#        }
#      }
#    }
#  }
#
#  death: __type(name: "MatchPlayerStatsDeathEventType") {
#    fields {
#      name
#      type {
#        kind
#        name
#        ofType {
#          kind
#          name
#        }
#      }
#    }
#  }
#
#  assist: __type(name: "MatchPlayerStatsAssistEventType") {
#    fields {
#      name
#      type {
#        kind
#        name
#        ofType {
#          kind
#          name
#        }
#      }
#    }
#  }
#}
#"""
#
#response_testing = requests.post(URL, headers=HEADERS, json={"query": query_testing})
#
#print("Status:", response_testing.status_code)
#print(response_testing.json(), " \n \n ")





#print("\n \n \n TESTING \n \n \n")
#
#def frage(query, variablen=None):
#    r = requests.post(URL, headers=HEADERS,
#                      json={"query": query, "variables": variablen or {}})
#    daten = r.json()
#    if "errors" in daten:
#        print(json.dumps(daten["errors"], indent=2))
#    return daten.get("data")
#
#
#def typ_als_text(t):
#    # Typen sind verschachtelt, z.B. "Liste von Int" oder "Pflichtfeld Long"
#    if t["kind"] == "NON_NULL":
#        return typ_als_text(t["ofType"]) + "!"
#    if t["kind"] == "LIST":
#        return "[" + typ_als_text(t["ofType"]) + "]"
#    return t["name"]
#
#daten = frage("""
#query {
#  __schema {
#    queryType {
#      fields { name type { name kind ofType { name } } }
#    }
#  }
#}
#""")
#for f in daten["__schema"]["queryType"]["fields"]:
#    if f["name"] == "match":
#        print(f["type"])
#
#
#
#def felder(typname):
#    daten = frage("""
#    query ($name: String!) {
#      __type(name: $name) {
#        fields {
#          name
#          type { name kind ofType { name kind ofType { name kind ofType { name kind } } } }
#        }
#      }
#    }
#    """, {"name": typname})
#
#    if not daten or not daten["__type"]:
#        print("Typ nicht gefunden:", typname)
#        return
#
#    for f in sorted(daten["__type"]["fields"], key=lambda x: x["name"]):
#        print(f"{f['name']:35} {typ_als_text(f['type'])}")
#
#
#felder("MatchType")




"""
actualRank                          Short
analysisOutcome                     MatchAnalysisOutcomeType
averageImp                          Short
averageRank                         Short
barracksStatusDire                  Short
barracksStatusRadiant               Short
bottomLaneOutcome                   LaneOutcomeEnums
bracket                             Byte
chatEvents                          [MatchStatsChatEventType]
clusterId                           Int
didRadiantWin                       Boolean
didRequestDownload                  Boolean
direKills                           [Int]
direTeam                            TeamType
direTeamId                          Int
durationSeconds                     Int
endDateTime                         Long
firstBloodTime                      Int
gameMode                            GameModeEnumType
gameVersionId                       Short
id                                  Long
isStats                             Boolean
laneReport                          MatchStatsLaneReportType
league                              LeagueType
leagueId                            Int
lobbyType                           LobbyTypeEnum
midLaneOutcome                      LaneOutcomeEnums
numHumanPlayers                     Int
parsedDateTime                      Long
pickBans                            [MatchStatsPickBanType]
playbackData                        MatchPlaybackDataType
players                             [MatchPlayerType]
predictedOutcomeWeight              Byte
predictedWinRates                   [Decimal]
radiantExperienceLeads              [Int]
radiantKills                        [Int]
radiantNetworthLeads                [Int]
radiantTeam                         TeamType
radiantTeamId                       Int
rank                                Int
regionId                            Byte
replaySalt                          Long
sequenceNum                         Long
series                              SeriesType
seriesId                            Long
spectators                          [MatchPlayerSpectatorType]
startDateTime                       Long
statsDateTime                       Long
topLaneOutcome                      LaneOutcomeEnums
tournamentId                        Int
tournamentRound                     Short
towerDeaths                         [MatchStatsTowerDeathType]
towerStatus                         [MatchStatsTowerReportType]
towerStatusDire                     Int
towerStatusRadiant                  Int
winRates                            [Decimal]
"""