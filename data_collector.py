import ast
import csv
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm




class Data_Collector:

    def __init__(self):
        """ Initialize variables to access the Stratz API. Currently, I am using my own steam account as reference for games."""
        self.API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJTdWJqZWN0IjoiMzliNTliNjUtNmI1OC00MGFmLWE0MzEtNDdmYWYxMWVhMDVlIiwiU3RlYW1JZCI6IjE0OTE5NDczOSIsIkFQSVVzZXIiOiJ0cnVlIiwibmJmIjoxNzkxMDIyOTg3LCJleHAiOjE4MjI1NTg5ODcsImlhdCI6MTc5MTAyMjk4NywiaXNzIjoiaHR0cHM6Ly9hcGkuc3RyYXR6LmNvbSJ9.fpm7gafT4TANe1my9oCTvN-nIz7pDGEvmyGgaflXBE4"
        self.URL = "https://api.stratz.com/graphql"
        self.HEADERS = {
            "Authorization": f"Bearer {self.API_KEY}",
            "User-Agent": "STRATZ_API",
            "Content-Type": "application/json",
        }
        self.STEAM_ID_64 = 76561198109460467  
        self.STEAM_ID = self.STEAM_ID_64 - 76561197960265728
        self.match_ids = []
        self.num_unaccessible_matches = 0
        self.unranked_matches = 0
        self.unsuitable_rank_matches = 0

    def get_single_match(self, match_id, time):
        """Extract all match data from a single match we use to train prediction models."""

        #try:
        # extract player-unspecific data from the match
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
        response_match = requests.post(self.URL, headers=self.HEADERS, json={"query": query_match})
        matchdata = response_match.json()

        # checking if there is accessible data (this does not mean there is content, i.e. data can still be None)
        try:
            _ = matchdata['data']
        except KeyError:
            self.num_unaccessible_matches += 1
            return [0]

        matchid = np.array(matchdata['data']['match']['id'])
        actualRank = np.array(int(matchdata['data']['match']['actualRank']))
        gameMode = np.array(matchdata['data']['match']['gameMode'])
        durationSeconds = np.array(int(matchdata['data']['match']['durationSeconds'])) # use to see how far we can go for this match: 0, 5, 10, 5, ... (max until 40 id say)
        didRadiantWin = np.array(matchdata['data']['match']['didRadiantWin'])
        startDateTime = np.array(matchdata['data']['match']['startDateTime'])

        # check that data is not empty (happens frequently for some statistics)
        if any(x is None for x in [matchid, actualRank, gameMode, durationSeconds, didRadiantWin, startDateTime]): 
            self.num_unaccessible_matches += 1
            return [0]
        
        # Check if match is suitable for analysis
        if int(actualRank) > 75 or int(actualRank) < 21:
            self.unsuitable_rank_matches += 1
            return [0] # ensure in methods calling working with output of this method that this case is covered

        if gameMode != "ALL_PICK_RANKED": # überlegen ob ich bei mehr matches auch ranked dann voraussetzen möchte
            self.unranked_matches += 1
            return [0] # ensure in methods calling working with output of this method that this case is covered

        # ... insert any additional code here to check for further desired characteristics of the match


        # ###########################################
        # extract player-specific data from the match

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

        response_playerstats = requests.post(self.URL, headers=self.HEADERS, json={"query": query_playerstats})
        playerdata = response_playerstats.json()
        playerdata = response_playerstats.json()
        #barracksStatusDire = np.array([int(x) for x in str(format(playerdata['data']['match']['barracksStatusDire'], "06b"))]) # vllt dann eher ein jnp.array später
        #barracksStatusRadiant = np.array([int(x) for x in str(format(playerdata['data']['match']['barracksStatusRadiant'], "06b"))])
        #towerStatusDire = np.array([int(x) for x in str(format(playerdata['data']['match']['towerStatusDire'], "011b"))])
        #towerStatusRadiant = np.array([int(x) for x in str(format(playerdata['data']['match']['towerStatusRadiant'], "011b"))])
        #towerDeaths = playerdata['data']['match']['towerDeaths']
        #towerStatus = playerdata['data']['match']['towerStatus']

        # computing the amount of time steps (5 mins each) we can get out of the game
        timesteps = durationSeconds // 60 // 5 +1 # +1 bc we also wann do it at 0
        timesteps = min(timesteps, 9)

        # check if matchdata is accessible (again this does not mean data cannot be None)
        try:
            _ = playerdata['data']['match']
        except KeyError:
            self.num_unaccessible_matches += 1
            return [0]

        if playerdata['data']['match']['direKills'] is None:
            self.num_unaccessible_matches += 1
            return [0]

        if playerdata['data']['match']['radiantKills'] is None:
            self.num_unaccessible_matches += 1
            return [0]
        
        if playerdata['data']['match']['radiantNetworthLeads'] is None:
            self.num_unaccessible_matches += 1
            return [0]

        if playerdata['data']['match']['radiantExperienceLeads'] is None:
            self.num_unaccessible_matches += 1
            return [0]

        direKills = np.array([np.sum(np.array(playerdata['data']['match']['direKills'], dtype=int)[:5*t + 1]) for t in range(timesteps)])
        radiantKills = np.array([np.sum(np.array(playerdata['data']['match']['radiantKills'], dtype=int)[:5*t + 1]) for t in range(timesteps)])
        radiantNetworthLeads = np.array(playerdata['data']['match']['radiantNetworthLeads'], dtype=int)[1::5]
        radiantExperienceLeads = np.array(playerdata['data']['match']['radiantExperienceLeads'], dtype=int)[1::5]

        players = playerdata['data']['match']['players']  
        if players is None:
            self.num_unaccessible_matches += 1
            return [0]
        isRadiant = np.array([player['isRadiant'] for player in players]) # danach kann man sie dann ordnen  
        if isRadiant is None:
            self.num_unaccessible_matches += 1
            return [0]
        heroIds = np.array([player['heroId'] for player in players]) 
        if heroIds is None:
            self.num_unaccessible_matches += 1
            return [0]
        level = [np.array(player['stats']['level']) for player in players]
        if level is None:
            self.num_unaccessible_matches += 1
            return [0]
        levels = np.array([np.array([np.sum(playerlevel < t*5*60) for t in range(timesteps)]) for playerlevel in level])
        lastHitsPerMinute = np.array([player['stats']['lastHitsPerMinute'] for player in players])
        if lastHitsPerMinute is None:
            self.num_unaccessible_matches += 1
            return [0]
        lastHits = np.array([np.array([np.sum(playerlhpm[:5*t]) for t in range(timesteps)]) for playerlhpm in lastHitsPerMinute])
        networthPerMinute = np.array([player['stats']['networthPerMinute'] for player in players])
        if networthPerMinute is None:
            self.num_unaccessible_matches += 1
            return [0]
        networths = np.array([[playernw[5*t] for t in range(timesteps)] for playernw in networthPerMinute])
        deniesPerMinute = np.array([player['stats']['deniesPerMinute'] for player in players])
        if deniesPerMinute is None:
            self.num_unaccessible_matches += 1
            return [0]
        denies = np.array([np.array([np.sum(playerdpm[:5*t]) for t in range(timesteps)]) for playerdpm in deniesPerMinute])
        killEvents = [player['stats']['killEvents'] for player in players]
        if killEvents is None:
            self.num_unaccessible_matches += 1
            return [0]
        killEvents = str(killEvents)
        killEvents = killEvents.replace("{'time': ", "").replace("}", "")
        killEvents = ast.literal_eval(killEvents)
        kills = np.array([np.array([np.sum(np.array(playerkill) < t*5*60) for t in range(timesteps)]) for playerkill in killEvents])
        deathEvents = [player['stats']['deathEvents'] for player in players]
        if deathEvents is None:
            self.num_unaccessible_matches += 1
            return [0]
        deathEvents = str(deathEvents)
        deathEvents = deathEvents.replace("{'time': ", "").replace("}", "")
        deathEvents = ast.literal_eval(deathEvents)
        deaths = np.array([np.array([np.sum(np.array(playerdeath) < t*5*60) for t in range(timesteps)]) for playerdeath in deathEvents])
        assistEvents = [player['stats']['assistEvents'] for player in players]
        if assistEvents is None:
            self.num_unaccessible_matches += 1
            return [0]
        assistEvents = str(assistEvents)
        assistEvents = assistEvents.replace("{'time': ", "").replace("}", "")
        assistEvents = ast.literal_eval(assistEvents)
        assists = np.array([np.array([np.sum(np.array(playerassist) < t*5*60) for t in range(timesteps)]) for playerassist in assistEvents])

        # jetzt hier einen output erzeugen, der jeweils alles zeitunabhängige vom match enthhält, und zusätzlich alles zeitabhängige match und player jeweils für einen Zeitpunkt (also insgesamt t+1 mal)
        # diesen code dann von anderem aus aufrufen, wo ich dann matches durchlaufe -> dafür in methode umschreiben (am besten so, dass man in methode zeit übergibt und dann hier nur der output für die zeit erzeugt wird)

        if time > timesteps:
            print(f"Time {time} is greater than available timesteps {timesteps}. Exiting.")
            return [0] # cover this case for functions/methods calling this method
        else:
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


    def get_match_ids(self, num_matches=100):
        """Extract all match ids from my num_matches recent games."""
        all_matches = []
        take = 50
        skip = 0
        while len(all_matches) < num_matches:
            query_matches = f"""
            query {{
                player(steamAccountId: {self.STEAM_ID}) {{
                    matches(request: {{ take: {take}, skip: {skip} }}) {{
                        id
                        startDateTime
                    }}
                }}
            }}
            """

            response = requests.post(
                self.URL,
                headers=self.HEADERS,
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

            if len(all_matches) %(num_matches//4) == 0:
                print(f"{len(all_matches)} Matches in total.")


        self.match_ids = [match["id"] for match in all_matches if match["id"] is not None]

        print("Newest Match-ID:", self.match_ids[0], "Date:", datetime.fromtimestamp(all_matches[0]["startDateTime"], tz=timezone.utc))
        print("Oldest Match-ID:", self.match_ids[-1], "Date:", datetime.fromtimestamp(all_matches[-1]["startDateTime"], tz=timezone.utc))


    def create_dataset_time(self, time):
        """Use match ids to create dataset using the get_single_match method."""
        dataset = [list(self.get_single_match(match_id, time)) for match_id in tqdm(self.match_ids, desc="Matches")]
        print(f"Number of unaccessible matches: {self.num_unaccessible_matches}.")
        print(f"Number of unranked matches: {self.unranked_matches}.")
        print(f"Number of unsuitable rank matches: {self.unsuitable_rank_matches}")
        dataset = [x for x in dataset if x != [0]]  
        print("Number of accessible matches: ", len(dataset))
        return dataset

    def store_dataset(self, dataset):
        """Store a dataset as a csv file."""
        # gucken, dass ich es als numpy file oder als csv store
        dataset_for_storing = [[x for sublist in group for x in sublist]for group in dataset]
        df = pd.DataFrame(dataset_for_storing,
                          columns=["match_id", "actualRank", "gameMode", "durationSeconds", "startDateTime", "didRadiantWin",  
                                   "direKills", "radiantKills", "radiantNetworthLeads", "radiantExperienceLeads",
                                   "heroId_1", "levels_1", "lastHits_1", "denies_1", "networth_1", "kills_1", "deaths_1", "assists_1",
                                   "heroId_2", "levels_2", "lastHits_2", "denies_2", "networth_2", "kills_2", "deaths_2", "assists_2",
                                   "heroId_3", "levels_3", "lastHits_3", "denies_3", "networth_3", "kills_3", "deaths_3", "assists_3",
                                   "heroId_4", "levels_4", "lastHits_4", "denies_4", "networth_4", "kills_4", "deaths_4", "assists_4",
                                   "heroId_5", "levels_5", "lastHits_5", "denies_5", "networth_5", "kills_5", "deaths_5", "assists_5",
                                   "heroId_6", "levels_6", "lastHits_6", "denies_6", "networth_6", "kills_6", "deaths_6", "assists_6",
                                   "heroId_7", "levels_7", "lastHits_7", "denies_7", "networth_7", "kills_7", "deaths_7", "assists_7",
                                   "heroId_8", "levels_8", "lastHits_8", "denies_8", "networth_8", "kills_8", "deaths_8", "assists_8",
                                   "heroId_9", "levels_9", "lastHits_9", "denies_9", "networth_9", "kills_9", "deaths_9", "assists_9",
                                   "heroId_10", "levels_10", "lastHits_10", "denies_10", "networth_10", "kills_10", "deaths_10", "assists_10",
                                   ])
        df.to_csv("dataset.csv", index=False)


    def preprocess_dataset(self):
        """Preprocess a dataset from storage."""
        # DATEN NOCH KONTROLLIEREN WEIL NACH DEM PP NATÜRLICH NICHTS MEHR ERKENNBAR
        with open("dataset.csv", "r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            next(reader)
            stored_dataset = list(reader)

        # bringing data into numpy compatible format
        stored_dataset = [ [ 1 if value == "True" else 0 if value == "False" else value for value in row] for row in stored_dataset ]
        stored_dataset = [ [ 1 if value == "ALL_PICK_RANKED" else value for value in row] for row in stored_dataset ]
        stored_dataset = [ [ int(value) for value in row] for row in stored_dataset ]
        stored_dataset = np.array(stored_dataset, dtype=np.float32)

        # shuffling the data
        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(len(stored_dataset))
        stored_dataset = stored_dataset[indices]
        n = len(stored_dataset)
        train_set = stored_dataset[:int(0.85 * n)]
        val_set = stored_dataset[int(0.85 * n):int(0.9 * n)]
        test_set = stored_dataset[int(0.9 * n):]

        # masking out data that doesn't need preprocessing
        needs_preprocessing = np.array([0, 0, 0, 0, 0, 0, 
                                         1, 1, 1, 1, 
                                         0, 1, 1, 1, 1, 1, 1, 1, 
                                         0, 1, 1, 1, 1, 1, 1, 1, 
                                         0, 1, 1, 1, 1, 1, 1, 1, 
                                         0, 1, 1, 1, 1, 1, 1, 1,  
                                         0, 1, 1, 1, 1, 1, 1, 1, 
                                         0, 1, 1, 1, 1, 1, 1, 1,
                                         0, 1, 1, 1, 1, 1, 1, 1,
                                         0, 1, 1, 1, 1, 1, 1, 1, 
                                         0, 1, 1, 1, 1, 1, 1, 1,
                                         0, 1, 1, 1, 1, 1, 1, 1,
                                         ]) 
        
        train_to_preprocess = train_set[:, needs_preprocessing.astype(bool)]
        val_to_preprocess = val_set[:, needs_preprocessing.astype(bool)]
        test_to_preprocess = test_set[:, needs_preprocessing.astype(bool)]
        
        mu_1 = np.mean(train_to_preprocess[:, :4], axis=0)
        sigma_1 = np.std(train_to_preprocess[:, :4], axis=0)
        mu_2 = np.mean(train_to_preprocess[:, 4:].reshape((-1, (train_to_preprocess.shape[1]-4)//10)), axis=0)
        sigma_2 = np.std(train_to_preprocess[:, 4:].reshape(-1, (train_to_preprocess.shape[1]-4)//10), axis=0)
        mu_2 = np.repeat(mu_2, 10)
        sigma_2 = np.repeat(sigma_2, 10)
        self.mu = np.concatenate((mu_1, mu_2))
        self.sigma = np.concatenate((sigma_1, sigma_2))
        self.sigma[np.isclose(self.sigma, 0)] = 1.0
        
        pp_train_data = (train_to_preprocess - self.mu[np.newaxis,:]) / self.sigma[np.newaxis,:]
        full_train_data = np.zeros_like(train_set)
        full_train_data[:, needs_preprocessing.astype(bool)] = pp_train_data
        full_train_data[:, ~needs_preprocessing.astype(bool)] = train_set[:, ~needs_preprocessing.astype(bool)]

        pp_val_data = (val_to_preprocess - self.mu[np.newaxis,:]) / self.sigma[np.newaxis,:]
        full_val_data = np.zeros_like(val_set)
        full_val_data[:, needs_preprocessing.astype(bool)] = pp_val_data
        full_val_data[:, ~needs_preprocessing.astype(bool)] = val_set[:, ~needs_preprocessing.astype(bool)]

        pp_test_data = (test_to_preprocess - self.mu[np.newaxis,:]) / self.sigma[np.newaxis,:]
        full_test_data = np.zeros_like(test_set)
        full_test_data[:, needs_preprocessing.astype(bool)] = pp_test_data
        full_test_data[:, ~needs_preprocessing.astype(bool)] = test_set[:, ~needs_preprocessing.astype(bool)]

        return full_train_data, full_val_data, full_test_data



"""
Overview of the data-structure we get per match (this is all for the specific time we set in create_dataset_time)

game_identification = [ matchid, actualRank, gameMode, durationSeconds, startDateTime ]
game_winner         = [ didRadiantWin ]
game_stats          = [ direKills, radiantKills, radiantNetworthLeads, radiantExperienceLeads ]
player1_Radiant     = [ heroId, levels, lastHits, denies, networth, kills, deaths, assists ]
player2_Radiant     = [ heroId, levels, lastHits, denies, networth, kills, deaths, assists ]
player3_Radiant     = [ heroId, levels, lastHits, denies, networth, kills, deaths, assists ]
player4_Radiant     = [ heroId, levels, lastHits, denies, networth, kills, deaths, assists ]
player5_Radiant     = [ heroId, levels, lastHits, denies, networth, kills, deaths, assists ]
player1_Dire        = [ heroId, levels, lastHits, denies, networth, kills, deaths, assists ]
player2_Dire        = [ heroId, levels, lastHits, denies, networth, kills, deaths, assists ]
player3_Dire        = [ heroId, levels, lastHits, denies, networth, kills, deaths, assists ]       
player4_Dire        = [ heroId, levels, lastHits, denies, networth, kills, deaths, assists ]
player5_Dire        = [ heroId, levels, lastHits, denies, networth, kills, deaths, assists ]

"""



# !take first 2k entries to have games of approx last X years and store them in array (2000xapprox 20 ) - train on 1200, validate on 100 and test on 200
# !preprocessing needs to be applied (CARE: NOT FOR EVERYTHING - e.g. not for hero id - also check other properties for their distribution bc outliers might kill the preprocessing)
# -> richtigkeit der daten gegen recent match crosschecken (einfach in dota gucken -> vor preprocessing natürlich)
# für preprocessen nur dei trainingsdaten nehmen




            # Further plans/ideas
            # PLAN: GLEICHES NETWORK AUF JEDEM PLAYER SEINE WERTE (DAVOR LERNBARES EMBEDDING AUF HERO ID JEWEILS) + NETWORK AUF CONCATENATED OUTPUTS DER RADIANT UND DIRE (GUCKEN, DASS DAS UNABHÄNGIG VON INPUT REIHENFOLGE DER SPIELER !!!) UND EIN NETWORK AUF MATCH SACHEN - UND DANN AUF DIE 3 CONCATENTED OUTPUT WIEDER EIN NETWORKT
            # lh und denies drin damit netzt sieht welche hero carry ist etc
            # didradiantwin brauch ich dann in game_stats noch dazu
            # tower und barack stats rauslassen, weil die sind nur ende des games - rausfinden ob die geparsed werden
            # bei netzwerken ein hero embedding trainieren (chatgpt fragen wie - weil one-hot denk zu viel)


if __name__ == "__main__":
    # hier noch ergänzen (mit Zeit 0 anfangen und dann kontrollieren)
    data_collector = Data_Collector()
    data_collector.get_match_ids(num_matches=50)
    dataset = data_collector.create_dataset_time(time=0)
    data_collector.store_dataset(dataset)
    data_collector.preprocess_dataset()

    # hier dann als nächstes pp

    # ->  gucken, ob matchstats wirklich bei so vielen fehlen, weil direKills etc eig bei vielen da - genauer untersuchen (wenn es nur an einer Sache scheitert lassen wir die halt raus)
    # -> noch fuer min 5 bis 40 validieren



# gucken, dass alle Minuten in etwa gleich viel Training Data habem
# MLPs jeweils auf einzelne Spieler anwenden, dann die outputs von radiant (irgendeine dim =! 1) zusammenfügen  und daruf mlp und fleiches mit dire und dann noch eins mit den allgemeinen stats und die outputs dieser 3 mlps wieder vereinen
# gucken, ob role immer s.d. 1 von jeder rolle (und nicht sowas wie roaming) und dann könnte man immer ein mlp für pos 1, eins für pos 2 und so weiter bei denen die anfangs auf spieler angewendetweren (sonst gleiches mlp auf alle andwenden, also einfach input in höhere dim transofrmieren (vmap netzwerk über die 5 input arrays, aber netzwerk für radiant und dire players kann ggf anderes sein, also 2 in total (+ die 3 die noch zusätzluch kommen)))
# kann man zur einfachheit anfangs aber auch weglassen aber auf keinen fall pro player ein anderers netzwerk dann
# in bits und dann onehotencoden


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