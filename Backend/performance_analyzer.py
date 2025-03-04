import asyncio
import os
import json
from datetime import datetime, timedelta
from termcolor import colored
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from video_database import VideoDatabase
from youtube_session_manager import YouTubeSessionManager
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PerformanceAnalyzer:
    def __init__(self):
        """Initialize the performance analyzer"""
        self.db = VideoDatabase()
        self.session_manager = YouTubeSessionManager()
        self.youtube_services = self.session_manager.initialize_all_sessions()
        
        # OpenAI API key for content optimization
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Output directory for reports
        self.reports_dir = "Backend/reports"
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Channels to analyze
        self.channels = [
            'tech_humor', 
            'ai_money', 
            'baby_tips', 
            'quick_meals', 
            'fitness_motivation'
        ]
    
    async def fetch_video_analytics(self, channel_type, days=30):
        """Fetch analytics for videos in a channel"""
        print(colored(f"\nFetching analytics for {channel_type}...", "blue"))
        
        try:
            youtube = self.youtube_services[channel_type]
            
            if not youtube:
                print(colored(f"✗ YouTube service not available for {channel_type}", "red"))
                return []
            
            # Get videos for channel
            self.db.connect()
            self.db.cursor.execute('''
            SELECT video_id, title, upload_date
            FROM videos
            WHERE channel_type = ? AND upload_date > ?
            ''', (
                channel_type,
                (datetime.now() - timedelta(days=days)).isoformat()
            ))
            
            videos = self.db.cursor.fetchall()
            self.db.disconnect()
            
            if not videos:
                print(colored(f"No videos found for {channel_type} in the last {days} days", "yellow"))
                return []
            
            # Get video IDs
            video_ids = [video[0] for video in videos]
            
            # Fetch analytics for videos
            analytics = []
            
            # Split video IDs into chunks of 50 (API limit)
            video_id_chunks = [video_ids[i:i+50] for i in range(0, len(video_ids), 50)]
            
            for chunk in video_id_chunks:
                # Get video statistics
                stats_response = youtube.videos().list(
                    part="statistics",
                    id=",".join(chunk)
                ).execute()
                
                for item in stats_response.get("items", []):
                    video_id = item["id"]
                    stats = item.get("statistics", {})
                    
                    # Find video details
                    video_details = next((v for v in videos if v[0] == video_id), None)
                    
                    if video_details:
                        title = video_details[1]
                        upload_date = video_details[2]
                        
                        # Create analytics entry
                        analytics.append({
                            'video_id': video_id,
                            'title': title,
                            'upload_date': upload_date,
                            'views': int(stats.get("viewCount", 0)),
                            'likes': int(stats.get("likeCount", 0)),
                            'comments': int(stats.get("commentCount", 0)),
                            'shares': 0  # YouTube API doesn't provide share count
                        })
            
            print(colored(f"✓ Fetched analytics for {len(analytics)} videos", "green"))
            return analytics
        
        except Exception as e:
            print(colored(f"✗ Error fetching analytics: {str(e)}", "red"))
            return []
    
    def update_performance_metrics(self, analytics):
        """Update performance metrics in the database"""
        if not analytics:
            return 0
        
        updated_count = 0
        
        for video in analytics:
            video_id = video['video_id']
            
            metrics = {
                'views': video['views'],
                'likes': video['likes'],
                'comments': video['comments'],
                'shares': video['shares'],
                'watch_time_seconds': 0,  # Not available from basic API
                'ctr': 0,  # Not available from basic API
                'avg_view_duration': 0  # Not available from basic API
            }
            
            success = self.db.update_performance(video_id, metrics)
            
            if success:
                updated_count += 1
        
        print(colored(f"✓ Updated performance metrics for {updated_count} videos", "green"))
        return updated_count
    
    def analyze_performance(self, channel_type, analytics):
        """Analyze performance of videos in a channel"""
        if not analytics:
            return None
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(analytics)
        
        # Calculate engagement rate (likes + comments) / views
        df['engagement_rate'] = (df['likes'] + df['comments']) / df['views'].apply(lambda x: max(x, 1))
        
        # Calculate like ratio
        df['like_ratio'] = df['likes'] / df['views'].apply(lambda x: max(x, 1))
        
        # Calculate comment ratio
        df['comment_ratio'] = df['comments'] / df['views'].apply(lambda x: max(x, 1))
        
        # Convert upload_date to datetime
        df['upload_date'] = pd.to_datetime(df['upload_date'])
        
        # Calculate days since upload
        df['days_since_upload'] = (datetime.now() - df['upload_date']).dt.days
        
        # Calculate views per day
        df['views_per_day'] = df['views'] / df['days_since_upload'].apply(lambda x: max(x, 1))
        
        # Sort by views (descending)
        df = df.sort_values('views', ascending=False)
        
        # Generate report
        report = {
            'channel_type': channel_type,
            'total_videos': len(df),
            'total_views': df['views'].sum(),
            'total_likes': df['likes'].sum(),
            'total_comments': df['comments'].sum(),
            'avg_views': df['views'].mean(),
            'avg_likes': df['likes'].mean(),
            'avg_comments': df['comments'].mean(),
            'avg_engagement_rate': df['engagement_rate'].mean(),
            'top_videos': df.head(5)[['video_id', 'title', 'views', 'likes', 'comments', 'engagement_rate']].to_dict('records'),
            'worst_videos': df.tail(5)[['video_id', 'title', 'views', 'likes', 'comments', 'engagement_rate']].to_dict('records')
        }
        
        return report, df
    
    def cluster_videos(self, df, n_clusters=3):
        """Cluster videos based on performance metrics"""
        if len(df) < n_clusters:
            return df
        
        # Select features for clustering
        features = ['views', 'likes', 'comments', 'engagement_rate', 'views_per_day']
        X = df[features].copy()
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Calculate cluster centers
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Create cluster summary
        cluster_summary = []
        
        for i in range(n_clusters):
            cluster_df = df[df['cluster'] == i]
            
            summary = {
                'cluster_id': i,
                'size': len(cluster_df),
                'avg_views': cluster_df['views'].mean(),
                'avg_likes': cluster_df['likes'].mean(),
                'avg_comments': cluster_df['comments'].mean(),
                'avg_engagement_rate': cluster_df['engagement_rate'].mean(),
                'top_videos': cluster_df.sort_values('views', ascending=False).head(3)[['video_id', 'title']].to_dict('records')
            }
            
            cluster_summary.append(summary)
        
        return df, cluster_summary
    
    def generate_performance_plots(self, channel_type, df):
        """Generate performance plots for a channel"""
        if df is None or len(df) == 0:
            return
        
        # Set plot style
        sns.set(style="whitegrid")
        
        # Create figure directory
        figures_dir = os.path.join(self.reports_dir, "figures", channel_type)
        os.makedirs(figures_dir, exist_ok=True)
        
        # 1. Views distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['views'], kde=True)
        plt.title(f'Views Distribution - {channel_type}')
        plt.xlabel('Views')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'views_distribution.png'))
        plt.close()
        
        # 2. Engagement metrics comparison
        plt.figure(figsize=(12, 6))
        
        # Create a new DataFrame with the top 10 videos by views
        top_df = df.sort_values('views', ascending=False).head(10).copy()
        
        # Melt the DataFrame for easier plotting
        melted_df = pd.melt(top_df, 
                           id_vars=['title'], 
                           value_vars=['views', 'likes', 'comments'],
                           var_name='Metric', value_name='Value')
        
        # Create the grouped bar chart
        chart = sns.catplot(x='title', y='Value', hue='Metric', data=melted_df, kind='bar', height=6, aspect=2)
        
        # Rotate x-axis labels
        chart.set_xticklabels(rotation=45, horizontalalignment='right')
        
        plt.title(f'Engagement Metrics - Top 10 Videos - {channel_type}')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'engagement_metrics.png'))
        plt.close()
        
        # 3. Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[['views', 'likes', 'comments', 'engagement_rate', 'views_per_day']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Correlation Heatmap - {channel_type}')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'correlation_heatmap.png'))
        plt.close()
        
        # 4. Views vs. Engagement Rate scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='views', y='engagement_rate', data=df)
        plt.title(f'Views vs. Engagement Rate - {channel_type}')
        plt.xlabel('Views')
        plt.ylabel('Engagement Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'views_vs_engagement.png'))
        plt.close()
        
        # 5. Time series of views
        if len(df) >= 5:
            plt.figure(figsize=(12, 6))
            time_df = df.sort_values('upload_date')
            plt.plot(time_df['upload_date'], time_df['views'], marker='o')
            plt.title(f'Views Over Time - {channel_type}')
            plt.xlabel('Upload Date')
            plt.ylabel('Views')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'views_over_time.png'))
            plt.close()
    
    async def generate_content_insights(self, channel_type, df):
        """Generate content insights using GPT"""
        if df is None or len(df) == 0 or not self.openai_api_key:
            return None
        
        try:
            # Prepare data for GPT
            top_videos = df.sort_values('views', ascending=False).head(5)[['title', 'views', 'likes', 'comments', 'engagement_rate']].to_dict('records')
            worst_videos = df.sort_values('views', ascending=True).head(5)[['title', 'views', 'likes', 'comments', 'engagement_rate']].to_dict('records')
            
            # Create prompt
            prompt = f"""
            Analyze the performance of YouTube Shorts videos for a {channel_type} channel.
            
            Top performing videos:
            {json.dumps(top_videos, indent=2)}
            
            Worst performing videos:
            {json.dumps(worst_videos, indent=2)}
            
            Based on this data, please provide:
            1. Key patterns in successful videos (topics, title structures, etc.)
            2. What to avoid based on poor performers
            3. Specific recommendations for future video topics
            4. Title structure suggestions
            5. Content strategy recommendations
            
            Format your response as JSON with the following structure:
            {{
                "patterns": ["pattern1", "pattern2", ...],
                "avoid": ["avoid1", "avoid2", ...],
                "topic_suggestions": ["topic1", "topic2", ...],
                "title_structures": ["structure1", "structure2", ...],
                "strategy_recommendations": ["recommendation1", "recommendation2", ...]
            }}
            """
            
            # Call GPT API
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a YouTube analytics expert who provides actionable insights for content creators."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                insights = json.loads(content)
                return insights
            except json.JSONDecodeError:
                # If not valid JSON, return the raw content
                return {"raw_insights": content}
        
        except Exception as e:
            print(colored(f"✗ Error generating content insights: {str(e)}", "red"))
            return None
    
    def generate_report(self, channel_type, report, df, cluster_summary=None, insights=None):
        """Generate a comprehensive performance report"""
        if report is None:
            return
        
        # Create report file
        report_file = os.path.join(self.reports_dir, f"{channel_type}_report.md")
        
        with open(report_file, 'w') as f:
            # Write header
            f.write(f"# Performance Report: {channel_type}\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write summary
            f.write("## Summary\n\n")
            f.write(f"Total Videos: {report['total_videos']}\n")
            f.write(f"Total Views: {report['total_views']}\n")
            f.write(f"Total Likes: {report['total_likes']}\n")
            f.write(f"Total Comments: {report['total_comments']}\n\n")
            
            f.write(f"Average Views: {report['avg_views']:.2f}\n")
            f.write(f"Average Likes: {report['avg_likes']:.2f}\n")
            f.write(f"Average Comments: {report['avg_comments']:.2f}\n")
            f.write(f"Average Engagement Rate: {report['avg_engagement_rate']:.2%}\n\n")
            
            # Write top videos
            f.write("## Top Performing Videos\n\n")
            for i, video in enumerate(report['top_videos']):
                f.write(f"{i+1}. **{video['title']}**\n")
                f.write(f"   - Views: {video['views']}\n")
                f.write(f"   - Likes: {video['likes']}\n")
                f.write(f"   - Comments: {video['comments']}\n")
                f.write(f"   - Engagement Rate: {video['engagement_rate']:.2%}\n\n")
            
            # Write worst videos
            f.write("## Lowest Performing Videos\n\n")
            for i, video in enumerate(report['worst_videos']):
                f.write(f"{i+1}. **{video['title']}**\n")
                f.write(f"   - Views: {video['views']}\n")
                f.write(f"   - Likes: {video['likes']}\n")
                f.write(f"   - Comments: {video['comments']}\n")
                f.write(f"   - Engagement Rate: {video['engagement_rate']:.2%}\n\n")
            
            # Write cluster analysis if available
            if cluster_summary:
                f.write("## Video Clusters\n\n")
                
                for cluster in cluster_summary:
                    f.write(f"### Cluster {cluster['cluster_id']+1}\n\n")
                    f.write(f"Size: {cluster['size']} videos\n")
                    f.write(f"Average Views: {cluster['avg_views']:.2f}\n")
                    f.write(f"Average Likes: {cluster['avg_likes']:.2f}\n")
                    f.write(f"Average Comments: {cluster['avg_comments']:.2f}\n")
                    f.write(f"Average Engagement Rate: {cluster['avg_engagement_rate']:.2%}\n\n")
                    
                    f.write("Top videos in this cluster:\n")
                    for i, video in enumerate(cluster['top_videos']):
                        f.write(f"{i+1}. {video['title']}\n")
                    
                    f.write("\n")
            
            # Write content insights if available
            if insights:
                f.write("## Content Insights\n\n")
                
                if "patterns" in insights:
                    f.write("### Patterns in Successful Videos\n\n")
                    for pattern in insights["patterns"]:
                        f.write(f"- {pattern}\n")
                    f.write("\n")
                
                if "avoid" in insights:
                    f.write("### What to Avoid\n\n")
                    for item in insights["avoid"]:
                        f.write(f"- {item}\n")
                    f.write("\n")
                
                if "topic_suggestions" in insights:
                    f.write("### Suggested Topics\n\n")
                    for topic in insights["topic_suggestions"]:
                        f.write(f"- {topic}\n")
                    f.write("\n")
                
                if "title_structures" in insights:
                    f.write("### Title Structure Suggestions\n\n")
                    for structure in insights["title_structures"]:
                        f.write(f"- {structure}\n")
                    f.write("\n")
                
                if "strategy_recommendations" in insights:
                    f.write("### Strategy Recommendations\n\n")
                    for recommendation in insights["strategy_recommendations"]:
                        f.write(f"- {recommendation}\n")
                    f.write("\n")
                
                if "raw_insights" in insights:
                    f.write("### Raw Insights\n\n")
                    f.write(insights["raw_insights"])
                    f.write("\n\n")
            
            # Write figures section
            f.write("## Performance Visualizations\n\n")
            f.write("See the 'figures' directory for visualizations of this channel's performance.\n\n")
            
            # Write conclusion
            f.write("## Conclusion\n\n")
            f.write("This report provides an overview of the channel's performance. Use these insights to guide your content strategy.\n")
            f.write("Regular analysis of performance metrics will help optimize your channel for growth.\n")
        
        print(colored(f"✓ Generated report for {channel_type}: {report_file}", "green"))
        return report_file
    
    async def analyze_channel(self, channel_type, days=30):
        """Analyze a single channel"""
        print(colored(f"\nAnalyzing {channel_type} channel...", "blue"))
        
        # Fetch analytics
        analytics = await self.fetch_video_analytics(channel_type, days)
        
        if not analytics:
            print(colored(f"No data available for {channel_type}", "yellow"))
            return
        
        # Update performance metrics
        self.update_performance_metrics(analytics)
        
        # Analyze performance
        report, df = self.analyze_performance(channel_type, analytics)
        
        if report is None:
            print(colored(f"Could not analyze performance for {channel_type}", "red"))
            return
        
        # Cluster videos
        if len(df) >= 3:
            df, cluster_summary = self.cluster_videos(df)
        else:
            cluster_summary = None
        
        # Generate plots
        self.generate_performance_plots(channel_type, df)
        
        # Generate content insights
        insights = await self.generate_content_insights(channel_type, df)
        
        # Generate report
        report_file = self.generate_report(channel_type, report, df, cluster_summary, insights)
        
        print(colored(f"✓ Analysis completed for {channel_type}", "green"))
        return report_file
    
    async def analyze_all_channels(self, days=30):
        """Analyze all channels"""
        print(colored("\nAnalyzing all channels...", "blue"))
        
        report_files = []
        
        for channel in self.channels:
            report_file = await self.analyze_channel(channel, days)
            
            if report_file:
                report_files.append(report_file)
        
        print(colored(f"✓ Analysis completed for all channels", "green"))
        return report_files

async def test_performance_analyzer():
    """Test the performance analyzer functionality"""
    analyzer = PerformanceAnalyzer()
    
    # Analyze all channels
    report_files = await analyzer.analyze_all_channels(days=30)
    
    if report_files:
        print(colored("\nGenerated reports:", "blue"))
        for report_file in report_files:
            print(colored(f"- {report_file}", "cyan"))
    else:
        print(colored("\nNo reports generated", "yellow"))

if __name__ == "__main__":
    asyncio.run(test_performance_analyzer()) 