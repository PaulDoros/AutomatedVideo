import asyncio
import os
import json
from datetime import datetime, timedelta
from termcolor import colored
import re
from video_database import VideoDatabase
from googleapiclient.discovery import build
from youtube_session_manager import YouTubeSessionManager
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ContentMonitor:
    def __init__(self):
        """Initialize the content monitor"""
        self.db = VideoDatabase()
        self.session_manager = YouTubeSessionManager()
        self.youtube_services = self.session_manager.initialize_all_sessions()
        
        # Minimum time gap before reusing a topic (in days)
        self.min_topic_gap = 60
        
        # Similarity threshold for content (0.0 to 1.0)
        # Higher values mean stricter uniqueness requirements
        self.similarity_threshold = 0.6
        
        # Initialize NLP components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
    
    async def fetch_channel_videos(self, channel_type, max_results=100):
        """Fetch videos from a YouTube channel"""
        print(colored(f"\nFetching videos for {channel_type}...", "blue"))
        
        try:
            youtube = self.youtube_services[channel_type]
            channel_id = self.session_manager.get_channel_id(channel_type)
            
            if not youtube or not channel_id:
                print(colored(f"✗ YouTube service or channel ID not available for {channel_type}", "red"))
                return []
            
            # Get uploads playlist ID
            channels_response = youtube.channels().list(
                part="contentDetails",
                id=channel_id
            ).execute()
            
            if not channels_response.get("items"):
                print(colored(f"✗ No channel found with ID {channel_id}", "red"))
                return []
            
            uploads_playlist_id = channels_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
            
            # Get videos from uploads playlist
            videos = []
            next_page_token = None
            
            while len(videos) < max_results:
                playlist_response = youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=uploads_playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                ).execute()
                
                videos.extend(playlist_response["items"])
                
                next_page_token = playlist_response.get("nextPageToken")
                if not next_page_token or len(videos) >= max_results:
                    break
            
            print(colored(f"✓ Fetched {len(videos)} videos for {channel_type}", "green"))
            return videos
        
        except Exception as e:
            print(colored(f"✗ Error fetching videos for {channel_type}: {str(e)}", "red"))
            return []
    
    async def fetch_video_statistics(self, channel_type, video_ids):
        """Fetch statistics for videos"""
        if not video_ids:
            return {}
        
        try:
            youtube = self.youtube_services[channel_type]
            
            if not youtube:
                print(colored(f"✗ YouTube service not available for {channel_type}", "red"))
                return {}
            
            # Split video IDs into chunks of 50 (API limit)
            video_id_chunks = [video_ids[i:i+50] for i in range(0, len(video_ids), 50)]
            
            all_stats = {}
            
            for chunk in video_id_chunks:
                stats_response = youtube.videos().list(
                    part="statistics",
                    id=",".join(chunk)
                ).execute()
                
                for item in stats_response.get("items", []):
                    video_id = item["id"]
                    stats = item.get("statistics", {})
                    all_stats[video_id] = stats
            
            return all_stats
        
        except Exception as e:
            print(colored(f"✗ Error fetching video statistics: {str(e)}", "red"))
            return {}
    
    def _preprocess_text(self, text):
        """Preprocess text for similarity comparison"""
        if not text:
            return ""
        
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in self.stop_words]
        
        return " ".join(tokens)
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Preprocess texts
        processed_text1 = self._preprocess_text(text1)
        processed_text2 = self._preprocess_text(text2)
        
        if not processed_text1 or not processed_text2:
            return 0.0
        
        # Vectorize texts
        try:
            tfidf_matrix = self.vectorizer.fit_transform([processed_text1, processed_text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception as e:
            print(colored(f"Error calculating similarity: {str(e)}", "red"))
            return 0.0
    
    def extract_keywords(self, text, top_n=10):
        """Extract keywords from text"""
        if not text:
            return []
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        if not processed_text:
            return []
        
        # Vectorize text
        try:
            tfidf_matrix = self.vectorizer.fit_transform([processed_text])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get feature weights
            dense = tfidf_matrix.todense()
            dense_list = dense.tolist()[0]
            
            # Create a list of (word, weight) pairs
            word_weights = [(feature_names[i], dense_list[i]) for i in range(len(feature_names))]
            
            # Sort by weight and take top N
            word_weights.sort(key=lambda x: x[1], reverse=True)
            keywords = [word for word, weight in word_weights[:top_n]]
            
            return keywords
        except Exception as e:
            print(colored(f"Error extracting keywords: {str(e)}", "red"))
            return []
    
    async def store_channel_videos(self, channel_type):
        """Store videos from a channel in the database"""
        videos = await self.fetch_channel_videos(channel_type)
        
        if not videos:
            return False
        
        # Get video IDs
        video_ids = [video["contentDetails"]["videoId"] for video in videos]
        
        # Fetch statistics for videos
        stats = await self.fetch_video_statistics(channel_type, video_ids)
        
        # Store videos in database
        stored_count = 0
        
        for video in videos:
            video_id = video["contentDetails"]["videoId"]
            snippet = video["snippet"]
            
            # Extract video data
            video_data = {
                'video_id': video_id,
                'channel_type': channel_type,
                'title': snippet.get("title", ""),
                'description': snippet.get("description", ""),
                'tags': snippet.get("tags", []),
                'upload_date': snippet.get("publishedAt", datetime.now().isoformat()),
                'status': 'published',
                'topic': snippet.get("title", "")  # Use title as topic
            }
            
            # Add to database
            success = self.db.add_video(video_data)
            
            if success:
                stored_count += 1
                
                # Extract keywords
                keywords = self.extract_keywords(snippet.get("title", "") + " " + snippet.get("description", ""))
                
                # Add to content history
                self.db.connect()
                self.db.cursor.execute('''
                INSERT INTO content_history (
                    channel_type, topic, keywords, upload_date, video_id
                ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    channel_type,
                    snippet.get("title", ""),
                    json.dumps(keywords),
                    snippet.get("publishedAt", datetime.now().isoformat()),
                    video_id
                ))
                self.db.conn.commit()
                self.db.disconnect()
                
                # Update performance metrics
                if video_id in stats:
                    metrics = {
                        'views': int(stats[video_id].get("viewCount", 0)),
                        'likes': int(stats[video_id].get("likeCount", 0)),
                        'comments': int(stats[video_id].get("commentCount", 0)),
                        'shares': 0  # YouTube API doesn't provide share count
                    }
                    
                    self.db.update_performance(video_id, metrics)
        
        print(colored(f"✓ Stored {stored_count} videos for {channel_type}", "green"))
        return stored_count > 0
    
    async def store_all_channel_videos(self):
        """Store videos from all channels"""
        print(colored("\nStoring videos from all channels...", "blue"))
        
        channels = list(self.youtube_services.keys())
        
        for channel in channels:
            await self.store_channel_videos(channel)
        
        print(colored("✓ Completed storing videos from all channels", "green"))
    
    def is_topic_duplicate(self, channel_type, topic):
        """Check if a topic is a duplicate"""
        # Check if topic exists in database
        if self.db.check_topic_exists(channel_type, topic, days=self.min_topic_gap):
            return True
        
        # Get recent topics for channel
        self.db.connect()
        self.db.cursor.execute('''
        SELECT topic FROM content_history
        WHERE channel_type = ? AND upload_date > ?
        ''', (
            channel_type,
            (datetime.now() - timedelta(days=self.min_topic_gap)).isoformat()
        ))
        
        recent_topics = [row[0] for row in self.db.cursor.fetchall()]
        self.db.disconnect()
        
        # Check similarity with recent topics
        for recent_topic in recent_topics:
            similarity = self.calculate_similarity(topic, recent_topic)
            
            if similarity > self.similarity_threshold:
                print(colored(f"Topic '{topic}' is similar to existing topic '{recent_topic}' (similarity: {similarity:.2f})", "yellow"))
                return True
        
        return False
    
    def is_content_duplicate(self, channel_type, content):
        """Check if content is a duplicate"""
        # Get recent content for channel
        self.db.connect()
        self.db.cursor.execute('''
        SELECT script FROM videos
        WHERE channel_type = ? AND upload_date > ?
        ''', (
            channel_type,
            (datetime.now() - timedelta(days=self.min_topic_gap)).isoformat()
        ))
        
        recent_scripts = [row[0] for row in self.db.cursor.fetchall() if row[0]]
        self.db.disconnect()
        
        # Check similarity with recent scripts
        for script in recent_scripts:
            similarity = self.calculate_similarity(content, script)
            
            if similarity > self.similarity_threshold:
                print(colored(f"Content has {similarity:.2f} similarity with existing content", "yellow"))
                return True
        
        return False
    
    def suggest_alternative_topic(self, channel_type, topic):
        """Suggest an alternative topic if the original is a duplicate"""
        # Extract keywords from topic
        keywords = self.extract_keywords(topic, top_n=5)
        
        if not keywords:
            return None
        
        # Get recent topics for channel
        self.db.connect()
        self.db.cursor.execute('''
        SELECT topic FROM content_history
        WHERE channel_type = ?
        ORDER BY upload_date DESC
        LIMIT 50
        ''', (channel_type,))
        
        recent_topics = [row[0] for row in self.db.cursor.fetchall()]
        self.db.disconnect()
        
        # Find topics that don't contain the keywords
        alternative_keywords = []
        
        for recent_topic in recent_topics:
            topic_keywords = self.extract_keywords(recent_topic, top_n=5)
            
            for keyword in topic_keywords:
                if keyword not in keywords and keyword not in alternative_keywords:
                    alternative_keywords.append(keyword)
        
        # Combine original and alternative keywords
        if alternative_keywords:
            # Take 2 original keywords and 3 alternative keywords
            combined_keywords = keywords[:2] + alternative_keywords[:3]
            
            # Create a new topic suggestion
            suggestion = f"Alternative for '{topic}': Try '{' '.join(combined_keywords)}'"
            return suggestion
        
        return None

async def test_content_monitor():
    """Test the content monitor functionality"""
    monitor = ContentMonitor()
    
    # Store videos from all channels
    await monitor.store_all_channel_videos()
    
    # Test topic duplication check
    test_topics = [
        "Make $100/Day with ChatGPT Automation",
        "Earn $100 Daily Using ChatGPT",  # Similar to the first
        "10-Minute Morning Workout Routine",
        "Quick 10-Minute Morning Exercise",  # Similar to the third
        "Completely Different Topic About Space Exploration"
    ]
    
    print(colored("\nTesting topic duplication check...", "blue"))
    
    for topic in test_topics:
        is_duplicate = monitor.is_topic_duplicate("tech_humor", topic)
        
        if is_duplicate:
            print(colored(f"✗ Topic '{topic}' is a duplicate", "yellow"))
            
            # Suggest alternative
            suggestion = monitor.suggest_alternative_topic("tech_humor", topic)
            
            if suggestion:
                print(colored(f"  Suggestion: {suggestion}", "cyan"))
        else:
            print(colored(f"✓ Topic '{topic}' is unique", "green"))
    
    print(colored("\nTest completed", "green"))

if __name__ == "__main__":
    asyncio.run(test_content_monitor()) 