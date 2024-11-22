def get_wikipedia_url(entity):
    url = 'https://wikidata.org/w/api.php'
    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": entity
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("search"):
                entity_id = data['search'][0]['id']
                details_url = 'https://www.wikidata.org/w/api.php'
                params = {
                    'action': 'wbgetentities',
                    'ids': entity_id,
                    'sites': 'enwiki',
                    'props': 'sitelinks',
                    'format': 'json',
                }
                entity_response = requests.get(details_url, params=params, timeout=5)
                entity_data = entity_response.json()
                if 'entities' in entity_data and entity_id in entity_data['entities']:
                    sitelinks = entity_data['entities'][entity_id].get('sitelinks', {})
                    if 'enwiki' in sitelinks:
                        wikipedia_title = sitelinks['enwiki']['title']
                        wikipedia_title = wikipedia_title.replace(' ', '_')
                        return f"https://en.wikipedia.org/wiki/{wikipedia_title}"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Wikipedia URL for {entity}: {e}")
    return None