import requests

from typing import List
from termcolor import colored

def search_for_stock_videos(query: str, api_key: str, it: int, min_dur: int) -> List[str]:
    """
    Searches for stock videos based on a query.
    Now optimized to find fewer but more relevant videos.
    """
    
    headers = {
        "Authorization": api_key
    }

    # Reduced number of results to process
    qurl = f"https://api.pexels.com/videos/search?query={query}&per_page={it}"

    r = requests.get(qurl, headers=headers)
    response = r.json()

    raw_urls = []
    video_url = None
    video_res = 0
    try:
        # Look for just one good quality video per query
        for i in range(it):
            if response["videos"][i]["duration"] >= min_dur:
                raw_urls = response["videos"][i]["video_files"]
                
                # Find the highest quality version
                for video in raw_urls:
                    if ".com/video-files" in video["link"]:
                        if (video["width"]*video["height"]) > video_res:
                            video_url = video["link"]
                            video_res = video["width"]*video["height"]
                
                if video_url:
                    break  # Stop once we find a good video

        # Let user know
        status = "found" if video_url else "no"
        print(colored(f"\t=> \"{query}\" {status} matching video", "cyan"))

        # Return single video URL or empty list
        return [video_url] if video_url else []

    except Exception as e:
        print(colored("[-] No Videos found.", "red"))
        print(colored(e, "red"))
        return []
