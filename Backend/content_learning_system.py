import os
import json
import time
from datetime import datetime, timedelta
import random
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict, Counter
from termcolor import colored
import re
import asyncio

# Ensure NLTK data is downloaded
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class ContentLearningSystem:
    """
    A self-learning system that analyzes content performance, prevents repetition,
    and suggests high-performing content ideas based on historical data.
    """
    
    def __init__(self):
        """Initialize the content learning system"""
        self.data_dir = "data/learning_system"
        self.performance_file = f"{self.data_dir}/content_performance.json"
        self.blacklist_file = f"{self.data_dir}/content_blacklist.json"
        self.suggestions_file = f"{self.data_dir}/content_suggestions.json"
        self.similarity_threshold = 0.75  # Threshold for content similarity
        self.max_history_items = 100  # Maximum number of items to keep in history
        self.min_performance_data_points = 5  # Minimum data points needed for reliable analysis
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize data structures
        self._load_data()
    
    def _load_data(self):
        """Load data from files or initialize if not exists"""
        # Performance data
        if os.path.exists(self.performance_file):
            try:
                with open(self.performance_file, 'r', encoding='utf-8') as f:
                    self.performance_data = json.load(f)
            except:
                self.performance_data = {"channels": {}}
        else:
            self.performance_data = {"channels": {}}
        
        # Blacklist data
        if os.path.exists(self.blacklist_file):
            try:
                with open(self.blacklist_file, 'r', encoding='utf-8') as f:
                    self.blacklist = json.load(f)
            except:
                self.blacklist = {"global": [], "channels": {}}
        else:
            self.blacklist = {"global": [], "channels": {}}
        
        # Suggestions data
        if os.path.exists(self.suggestions_file):
            try:
                with open(self.suggestions_file, 'r', encoding='utf-8') as f:
                    self.suggestions = json.load(f)
            except:
                self.suggestions = {"channels": {}}
        else:
            self.suggestions = {"channels": {}}
        
        # Initialize channel data if not exists
        for channel in ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation']:
            if channel not in self.performance_data["channels"]:
                self.performance_data["channels"][channel] = {
                    "videos": [],
                    "keywords": {},
                    "topics": {},
                    "last_updated": datetime.now().isoformat()
                }
            
            if channel not in self.blacklist["channels"]:
                self.blacklist["channels"][channel] = []
            
            if channel not in self.suggestions["channels"]:
                self.suggestions["channels"][channel] = []
    
    def _save_data(self):
        """Save data to files"""
        # Save performance data
        with open(self.performance_file, 'w', encoding='utf-8') as f:
            json.dump(self.performance_data, f, indent=2)
        
        # Save blacklist data
        with open(self.blacklist_file, 'w', encoding='utf-8') as f:
            json.dump(self.blacklist, f, indent=2)
        
        # Save suggestions data
        with open(self.suggestions_file, 'w', encoding='utf-8') as f:
            json.dump(self.suggestions, f, indent=2)
    
    def _extract_keywords(self, text):
        """Extract keywords from text"""
        # Simple keyword extraction using NLTK
        words = nltk.word_tokenize(text.lower())
        stopwords = nltk.corpus.stopwords.words('english')
        words = [word for word in words if word.isalnum() and word not in stopwords and len(word) > 3]
        return Counter(words)
    
    def _calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts using TF-IDF and cosine similarity"""
        if not text1 or not text2:
            return 0.0
        
        try:
            vectorizer = TfidfVectorizer().fit_transform([text1, text2])
            vectors = vectorizer.toarray()
            return cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        except:
            # Fallback to simple word overlap if vectorization fails
            words1 = set(nltk.word_tokenize(text1.lower()))
            words2 = set(nltk.word_tokenize(text2.lower()))
            if not words1 or not words2:
                return 0.0
            return len(words1.intersection(words2)) / max(len(words1), len(words2))
    
    def record_content_performance(self, video_id, channel, content, metrics):
        """Record performance metrics for a piece of content"""
        if channel not in self.performance_data["channels"]:
            self.performance_data["channels"][channel] = {
                "videos": [],
                "keywords": {},
                "topics": {},
                "last_updated": datetime.now().isoformat()
            }
        
        # Extract keywords
        keywords = self._extract_keywords(content)
        
        # Create content record
        record = {
            "video_id": video_id,
            "content": content,
            "metrics": metrics,
            "keywords": dict(keywords.most_common(10)),
            "timestamp": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Add to videos list
        self.performance_data["channels"][channel]["videos"].append(record)
        
        # Update keywords performance
        for keyword, count in keywords.items():
            if keyword not in self.performance_data["channels"][channel]["keywords"]:
                self.performance_data["channels"][channel]["keywords"][keyword] = {
                    "count": 0,
                    "videos": [],
                    "avg_views": 0,
                    "avg_likes": 0,
                    "avg_comments": 0
                }
            
            self.performance_data["channels"][channel]["keywords"][keyword]["count"] += count
            self.performance_data["channels"][channel]["keywords"][keyword]["videos"].append(video_id)
        
        # Limit history size
        if len(self.performance_data["channels"][channel]["videos"]) > self.max_history_items:
            self.performance_data["channels"][channel]["videos"] = self.performance_data["channels"][channel]["videos"][-self.max_history_items:]
        
        # Update last updated timestamp
        self.performance_data["channels"][channel]["last_updated"] = datetime.now().isoformat()
        
        # Save data
        self._save_data()
        
        return True
    
    async def update_performance_data(self, video_id, channel, metrics):
        """Update performance metrics for a video"""
        if channel not in self.performance_data["channels"]:
            return False
        
        # Find video in data
        for video in self.performance_data["channels"][channel]["videos"]:
            if video["video_id"] == video_id:
                # Update metrics
                video["metrics"] = metrics
                video["last_updated"] = datetime.now().isoformat()
                
                # Update keywords performance
                for keyword in video["keywords"]:
                    if keyword in self.performance_data["channels"][channel]["keywords"]:
                        # Recalculate averages
                        keyword_data = self.performance_data["channels"][channel]["keywords"][keyword]
                        videos_with_keyword = [v for v in self.performance_data["channels"][channel]["videos"] 
                                              if v["video_id"] in keyword_data["videos"]]
                        
                        if videos_with_keyword:
                            keyword_data["avg_views"] = sum(v["metrics"]["views"] for v in videos_with_keyword) / len(videos_with_keyword)
                            keyword_data["avg_likes"] = sum(v["metrics"]["likes"] for v in videos_with_keyword) / len(videos_with_keyword)
                            keyword_data["avg_comments"] = sum(v["metrics"]["comments"] for v in videos_with_keyword) / len(videos_with_keyword)
                
                # Save data
                self._save_data()
                return True
        
        return False
    
    async def update_all_performance_data(self):
        """Update performance data for all videos using the PerformanceAnalyzer"""
        try:
            from performance_analyzer import PerformanceAnalyzer
            analyzer = PerformanceAnalyzer()
            
            # Get performance data for all channels
            for channel in self.performance_data["channels"]:
                # Get videos for this channel
                videos = self.performance_data["channels"][channel]["videos"]
                
                for video in videos:
                    video_id = video.get("video_id")
                    if video_id:
                        # Get metrics from analyzer
                        metrics = await analyzer.get_video_metrics(video_id)
                        if metrics:
                            # Update metrics
                            await self.update_performance_data(video_id, channel, metrics)
            
            return True
        except Exception as e:
            print(colored(f"Error updating performance data: {str(e)}", "red"))
            return False
    
    def is_content_repetitive(self, channel, content):
        """Check if content is too similar to recent content"""
        if channel not in self.performance_data["channels"]:
            return False, {"message": "Channel not found"}
        
        # Get recent videos
        recent_videos = self.performance_data["channels"][channel]["videos"][-10:]
        
        for video in recent_videos:
            similarity = self._calculate_similarity(content, video["content"])
            if similarity > self.similarity_threshold:
                return True, {
                    "message": f"Content is {similarity:.2%} similar to a recent video",
                    "similar_video_id": video["video_id"],
                    "similarity": similarity
                }
        
        return False, {"message": "Content is unique"}
    
    def is_content_blacklisted(self, channel, content):
        """Check if content contains blacklisted terms"""
        # Check global blacklist
        for item in self.blacklist["global"]:
            if item.lower() in content.lower():
                return True, f"Contains globally blacklisted term: {item}"
        
        # Check channel-specific blacklist
        if channel in self.blacklist["channels"]:
            for item in self.blacklist["channels"][channel]:
                if item.lower() in content.lower():
                    return True, f"Contains channel-specific blacklisted term: {item}"
        
        return False, ""
    
    def blacklist_content(self, channel, term, is_global=False):
        """Add a term to the blacklist"""
        if is_global:
            if term not in self.blacklist["global"]:
                self.blacklist["global"].append(term)
        else:
            if channel not in self.blacklist["channels"]:
                self.blacklist["channels"][channel] = []
            
            if term not in self.blacklist["channels"][channel]:
                self.blacklist["channels"][channel].append(term)
        
        # Save data
        self._save_data()
        return True
    
    def get_content_suggestions(self, channel, topic=None):
        """Get content suggestions based on performance data"""
        if channel not in self.performance_data["channels"]:
            return []
        
        suggestions = []
        channel_data = self.performance_data["channels"][channel]
        
        # Check if we have enough data
        if len(channel_data["videos"]) < self.min_performance_data_points:
            # Not enough data, return basic suggestions
            if channel == "tech_humor":
                suggestions.append({
                    "type": "content",
                    "title": "When programmers try to fix bugs at 3 AM",
                    "message": "Content about late-night debugging struggles tends to be relatable",
                    "score": 0.7
                })
            elif channel == "ai_money":
                suggestions.append({
                    "type": "content",
                    "title": "How to make $100/day with AI tools",
                    "message": "Content about making money with AI tends to perform well",
                    "score": 0.7
                })
            return suggestions
        
        # Find top performing keywords
        keyword_performance = []
        for keyword, data in channel_data["keywords"].items():
            if len(data["videos"]) >= 2:  # At least 2 videos with this keyword
                score = (data["avg_views"] + data["avg_likes"] * 10 + data["avg_comments"] * 20) / 100
                keyword_performance.append((keyword, score))
        
        # Sort by performance score
        keyword_performance.sort(key=lambda x: x[1], reverse=True)
        top_keywords = keyword_performance[:10]
        
        # Generate suggestions based on top keywords
        for keyword, score in top_keywords:
            # Skip if score is too low
            if score < 0.5:
                continue
            
            # Generate title based on keyword and channel
            if channel == "tech_humor":
                title = f"When {keyword} goes hilariously wrong"
                message = f"Content about '{keyword}' has performed well with {score:.1f} engagement score"
            elif channel == "ai_money":
                title = f"Make money with {keyword} automation"
                message = f"Content about '{keyword}' has performed well with {score:.1f} engagement score"
            elif channel == "baby_tips":
                title = f"Essential {keyword} tips for new parents"
                message = f"Content about '{keyword}' has performed well with {score:.1f} engagement score"
            elif channel == "quick_meals":
                title = f"5-minute {keyword} recipe anyone can make"
                message = f"Content about '{keyword}' has performed well with {score:.1f} engagement score"
            elif channel == "fitness_motivation":
                title = f"{keyword} workout for quick results"
                message = f"Content about '{keyword}' has performed well with {score:.1f} engagement score"
            else:
                title = f"{keyword} tips and tricks"
                message = f"Content about '{keyword}' has performed well with {score:.1f} engagement score"
            
            suggestions.append({
                "type": "content",
                "title": title,
                "message": message,
                "score": score,
                "keyword": keyword
            })
        
        # If topic is provided, filter or adapt suggestions
        if topic:
            # Check if topic contains any of our top keywords
            topic_keywords = self._extract_keywords(topic)
            matching_suggestions = []
            
            for suggestion in suggestions:
                if suggestion["keyword"] in topic_keywords:
                    matching_suggestions.append(suggestion)
            
            if matching_suggestions:
                return matching_suggestions
            
            # If no direct matches, find related suggestions
            related_suggestions = []
            for suggestion in suggestions:
                similarity = self._calculate_similarity(topic, suggestion["title"])
                if similarity > 0.3:  # Lower threshold for topic matching
                    suggestion["score"] *= similarity  # Adjust score based on relevance
                    related_suggestions.append(suggestion)
            
            if related_suggestions:
                return sorted(related_suggestions, key=lambda x: x["score"], reverse=True)
        
        # Return top suggestions
        return sorted(suggestions, key=lambda x: x["score"], reverse=True)[:5]
    
    def analyze_channel_content(self, channel):
        """Analyze content patterns for a channel"""
        if channel not in self.performance_data["channels"]:
            return {"status": "error", "message": "Channel not found"}
        
        channel_data = self.performance_data["channels"][channel]
        videos = channel_data["videos"]
        
        if len(videos) < self.min_performance_data_points:
            return {
                "status": "warning",
                "message": f"Not enough data (need at least {self.min_performance_data_points} videos)",
                "video_count": len(videos)
            }
        
        # Calculate average metrics
        avg_views = sum(v["metrics"].get("views", 0) for v in videos) / len(videos)
        avg_likes = sum(v["metrics"].get("likes", 0) for v in videos) / len(videos)
        avg_comments = sum(v["metrics"].get("comments", 0) for v in videos) / len(videos)
        
        # Extract all keywords
        all_keywords = Counter()
        for video in videos:
            keywords = video.get("keywords", {})
            for keyword, count in keywords.items():
                all_keywords[keyword] += count
        
        # Find content clusters using similarity
        clusters = []
        unclustered_videos = videos.copy()
        
        while unclustered_videos:
            current_video = unclustered_videos.pop(0)
            current_cluster = [current_video]
            
            i = 0
            while i < len(unclustered_videos):
                similarity = self._calculate_similarity(current_video["content"], unclustered_videos[i]["content"])
                if similarity > 0.6:  # Lower threshold for clustering
                    current_cluster.append(unclustered_videos.pop(i))
                else:
                    i += 1
            
            clusters.append(current_cluster)
        
        # Calculate cluster metrics
        cluster_data = []
        for i, cluster in enumerate(clusters):
            cluster_keywords = Counter()
            for video in cluster:
                for keyword, count in video.get("keywords", {}).items():
                    cluster_keywords[keyword] += count
            
            avg_cluster_views = sum(v["metrics"].get("views", 0) for v in cluster) / len(cluster)
            
            cluster_data.append({
                "id": i,
                "size": len(cluster),
                "avg_views": avg_cluster_views,
                "top_keywords": cluster_keywords.most_common(5),
                "performance_ratio": avg_cluster_views / avg_views if avg_views > 0 else 0
            })
        
        # Sort clusters by performance
        cluster_data.sort(key=lambda x: x["performance_ratio"], reverse=True)
        
        return {
            "status": "success",
            "video_count": len(videos),
            "avg_metrics": {
                "views": avg_views,
                "likes": avg_likes,
                "comments": avg_comments
            },
            "top_keywords": all_keywords.most_common(20),
            "content_clusters": cluster_data
        }
    
    def get_content_diversity_score(self, channel):
        """Calculate content diversity score for a channel (0-1)"""
        if channel not in self.performance_data["channels"]:
            return 0.0
        
        channel_data = self.performance_data["channels"][channel]
        videos = channel_data["videos"]
        
        if len(videos) < 3:
            return 0.5  # Default score for channels with few videos
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(videos)):
            for j in range(i+1, len(videos)):
                similarity = self._calculate_similarity(videos[i]["content"], videos[j]["content"])
                similarities.append(similarity)
        
        if not similarities:
            return 0.5
        
        # Average similarity (lower is more diverse)
        avg_similarity = sum(similarities) / len(similarities)
        
        # Convert to diversity score (1 - similarity)
        diversity_score = 1 - avg_similarity
        
        return diversity_score
    
    def add_content_suggestion(self, channel, suggestion):
        """Add a manual content suggestion"""
        if channel not in self.suggestions["channels"]:
            self.suggestions["channels"][channel] = []
        
        # Add suggestion with timestamp
        suggestion["timestamp"] = datetime.now().isoformat()
        self.suggestions["channels"][channel].append(suggestion)
        
        # Limit suggestions list size
        if len(self.suggestions["channels"][channel]) > 20:
            self.suggestions["channels"][channel] = self.suggestions["channels"][channel][-20:]
        
        # Save data
        self._save_data()
        return True 