import requests

# URL du endpoint pour les arrivées
url = "https://opensky-network.org/api/flights/arrival"

# Paramètres de la requête (timestamps UNIX en secondes)
params = {
    'airport': 'KJFK',
    'begin': 1609459200,  # Timestamp UNIX en secondes
    'end': 1609545600     # Timestamp UNIX en secondes
}

# Envoyer la requête GET
response = requests.get(url, params=params)

# Vérifier la réponse
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Erreur : {response.status_code} - {response.text}")
