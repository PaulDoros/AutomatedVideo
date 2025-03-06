import requests
import random
import os
import time
from typing import List, Dict, Any, Optional
from termcolor import colored
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def search_for_stock_videos(query: str, api_key: str, it: int = 5, min_dur: int = 10) -> List[str]:
    """
    Searches for high-quality, relevant stock videos based on a query.
    Enhanced to find more relevant videos with better quality and diversity.
    
    Args:
        query: Search query
        api_key: Pexels API key
        it: Number of videos to return
        min_dur: Minimum duration in seconds
        
    Returns:
        List of video URLs
    """
    try:
        # Enhance query with relevant keywords for better results
        enhanced_query = enhance_search_query(query)
        print(colored(f"Enhanced search query: '{enhanced_query}'", "blue"))
        
        headers = {
            "Authorization": api_key
        }

        # Request more results to have a better selection
        per_page = min(25, it * 3)  # Request more than needed to filter for quality
        qurl = f"https://api.pexels.com/videos/search?query={enhanced_query}&per_page={per_page}&orientation=portrait"

        # Make the API request
        response = requests.get(qurl, headers=headers)
        
        # Check if the request was successful
        if response.status_code != 200:
            print(colored(f"API request failed with status code {response.status_code}", "red"))
            print(colored(f"Response: {response.text}", "red"))
            return []
            
        data = response.json()
        
        # Check if we got any videos
        if "videos" not in data or not data["videos"]:
            print(colored(f"No videos found for query: '{enhanced_query}'", "yellow"))
            # Try a more generic query as fallback
            if query != enhanced_query and "videos" not in data:
                print(colored(f"Trying original query: '{query}'", "yellow"))
                return search_for_stock_videos(query, api_key, it, min_dur)
            return []

        # Process and filter videos
        videos = data["videos"]
        print(colored(f"Found {len(videos)} videos for query: '{enhanced_query}'", "green"))
        
        # Filter videos by duration
        valid_videos = [v for v in videos if v["duration"] >= min_dur]
        
        if not valid_videos:
            print(colored(f"No videos with minimum duration of {min_dur}s found", "yellow"))
            # Try with shorter duration as fallback
            shorter_min_dur = max(5, min_dur // 2)
            print(colored(f"Trying with shorter minimum duration: {shorter_min_dur}s", "yellow"))
            valid_videos = [v for v in videos if v["duration"] >= shorter_min_dur]
            
            if not valid_videos:
                print(colored("Still no valid videos found", "red"))
                return []
        
        # Sort videos by quality score (combination of resolution, duration, and relevance)
        scored_videos = []
        for video in valid_videos:
            # Find the highest quality version
            best_file = get_best_quality_file(video["video_files"])
            if best_file:
                # Calculate quality score
                resolution = best_file["width"] * best_file["height"]
                quality_score = (resolution / 1000000) * min(2, video["duration"] / min_dur)
                
                # Add relevance boost if query terms appear in video tags or user name
                if "user" in video and "name" in video["user"]:
                    video_tags = " ".join(video.get("tags", [])).lower()
                    user_name = video["user"]["name"].lower()
                    query_terms = query.lower().split()
                    
                    for term in query_terms:
                        if term in video_tags:
                            quality_score *= 1.5  # 50% boost for tag match
                        if term in user_name:
                            quality_score *= 1.2  # 20% boost for user match
                
                scored_videos.append({
                    "video": video,
                    "best_file": best_file,
                    "score": quality_score
                })
        
        # Sort by quality score (highest first)
        scored_videos.sort(key=lambda x: x["score"], reverse=True)
        
        # Select top videos, but ensure diversity by not taking all from the same creator
        selected_videos = []
        seen_creators = set()
        
        # First pass: take top videos from different creators
        for video_data in scored_videos:
            video = video_data["video"]
            best_file = video_data["best_file"]
            
            # Skip if we already have enough videos
            if len(selected_videos) >= it:
                break
                
            # Get creator ID
            creator_id = video["user"]["id"] if "user" in video and "id" in video["user"] else None
            
            # Skip if we already have a video from this creator (unless we're running out of options)
            if creator_id and creator_id in seen_creators and len(scored_videos) > it * 1.5:
                continue
                
            # Add to selected videos
            selected_videos.append(best_file["link"])
            if creator_id:
                seen_creators.add(creator_id)
        
        # Second pass: fill remaining slots if needed
        if len(selected_videos) < it:
            remaining_slots = it - len(selected_videos)
            for video_data in scored_videos:
                if video_data["best_file"]["link"] not in selected_videos:
                    selected_videos.append(video_data["best_file"]["link"])
                    if len(selected_videos) >= it:
                        break
        
        # Print results
        print(colored(f"Selected {len(selected_videos)} videos for query: '{enhanced_query}'", "green"))
        for i, url in enumerate(selected_videos):
            print(colored(f"  {i+1}. {url}", "cyan"))
            
        return selected_videos

    except Exception as e:
        print(colored(f"Error searching for videos: {str(e)}", "red"))
        return []

def get_best_quality_file(video_files: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Find the best quality video file from a list of video files.
    
    Args:
        video_files: List of video file dictionaries from Pexels API
        
    Returns:
        Best quality video file dictionary or None if no valid files
    """
    # Filter for valid video files (must have width, height, and link)
    valid_files = [
        f for f in video_files 
        if "width" in f and "height" in f and "link" in f and ".com/video-files" in f["link"]
    ]
    
    if not valid_files:
        return None
        
    # Sort by resolution (highest first)
    valid_files.sort(key=lambda x: x["width"] * x["height"], reverse=True)
    
    # Return the highest resolution file
    return valid_files[0]

def enhance_search_query(query: str) -> str:
    """
    Enhance a search query to get better video results.
    
    Args:
        query: Original search query
        
    Returns:
        Enhanced search query
    """
    # List of enhancement keywords for different types of content
    enhancements = {
        "tech": ["technology", "digital", "computer", "innovation", "futuristic"],
        "money": ["finance", "business", "success", "investment", "wealth"],
        "ai": ["artificial intelligence", "machine learning", "robot", "automation", "digital"],
        "baby": ["infant", "child", "family", "parenting", "cute"],
        "food": ["cooking", "meal", "kitchen", "delicious", "recipe"],
        "fitness": ["workout", "exercise", "gym", "healthy", "training"]
    }
    
    # Check if query contains any of the enhancement keywords
    query_lower = query.lower()
    matched_categories = []
    
    for category, keywords in enhancements.items():
        if category in query_lower or any(keyword in query_lower for keyword in keywords):
            matched_categories.append(category)
    
    # If no categories matched, return the original query
    if not matched_categories:
        return query
        
    # Select a random enhancement keyword from each matched category
    enhancement_keywords = []
    for category in matched_categories:
        keyword = random.choice(enhancements[category])
        if keyword not in query_lower:
            enhancement_keywords.append(keyword)
    
    # Add quality keywords
    quality_keywords = ["high quality", "professional", "cinematic", "beautiful"]
    enhancement_keywords.append(random.choice(quality_keywords))
    
    # Combine original query with enhancement keywords
    enhanced_query = query
    if enhancement_keywords:
        # Add up to 2 enhancement keywords to avoid diluting the search too much
        selected_enhancements = random.sample(enhancement_keywords, min(2, len(enhancement_keywords)))
        enhanced_query = f"{query} {' '.join(selected_enhancements)}"
    
    return enhanced_query
