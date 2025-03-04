import sqlite3
import os
import json
from datetime import datetime, timedelta
import pandas as pd
from termcolor import colored

class VideoDatabase:
    def __init__(self, db_path="Backend/data/video_database.sqlite"):
        """Initialize the video database"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.initialize_database()
    
    def connect(self):
        """Connect to the database"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
    def disconnect(self):
        """Disconnect from the database"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def initialize_database(self):
        """Create database tables if they don't exist"""
        self.connect()
        
        # Videos table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT UNIQUE,
            channel_type TEXT,
            title TEXT,
            description TEXT,
            tags TEXT,
            upload_date TIMESTAMP,
            scheduled_time TIMESTAMP,
            status TEXT,
            file_path TEXT,
            thumbnail_path TEXT,
            topic TEXT,
            script TEXT
        )
        ''')
        
        # Performance metrics table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT UNIQUE,
            views INTEGER DEFAULT 0,
            likes INTEGER DEFAULT 0,
            comments INTEGER DEFAULT 0,
            shares INTEGER DEFAULT 0,
            watch_time_seconds INTEGER DEFAULT 0,
            ctr REAL DEFAULT 0,
            avg_view_duration REAL DEFAULT 0,
            last_updated TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES videos(video_id)
        )
        ''')
        
        # Scheduling table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_type TEXT,
            scheduled_time TIMESTAMP,
            status TEXT DEFAULT 'pending',
            video_id TEXT,
            FOREIGN KEY (video_id) REFERENCES videos(video_id)
        )
        ''')
        
        # Content history table for tracking topics
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS content_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_type TEXT,
            topic TEXT,
            keywords TEXT,
            upload_date TIMESTAMP,
            video_id TEXT,
            FOREIGN KEY (video_id) REFERENCES videos(video_id)
        )
        ''')
        
        # Upload logs table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS upload_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            timestamp TIMESTAMP,
            action TEXT,
            status TEXT,
            message TEXT,
            FOREIGN KEY (video_id) REFERENCES videos(video_id)
        )
        ''')
        
        self.conn.commit()
        self.disconnect()
    
    def add_video(self, video_data):
        """Add a new video to the database"""
        self.connect()
        
        try:
            # Convert tags list to JSON string
            if 'tags' in video_data and isinstance(video_data['tags'], list):
                video_data['tags'] = json.dumps(video_data['tags'])
            
            # Insert video data
            self.cursor.execute('''
            INSERT INTO videos (
                video_id, channel_type, title, description, tags, 
                upload_date, scheduled_time, status, file_path, 
                thumbnail_path, topic, script
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                video_data.get('video_id', ''),
                video_data.get('channel_type', ''),
                video_data.get('title', ''),
                video_data.get('description', ''),
                video_data.get('tags', ''),
                video_data.get('upload_date', datetime.now()),
                video_data.get('scheduled_time', None),
                video_data.get('status', 'pending'),
                video_data.get('file_path', ''),
                video_data.get('thumbnail_path', ''),
                video_data.get('topic', ''),
                video_data.get('script', '')
            ))
            
            # Initialize performance entry
            if video_data.get('video_id'):
                self.cursor.execute('''
                INSERT INTO performance (video_id, last_updated)
                VALUES (?, ?)
                ''', (video_data.get('video_id', ''), datetime.now()))
                
            # Add to content history
            if video_data.get('topic'):
                self.cursor.execute('''
                INSERT INTO content_history (
                    channel_type, topic, keywords, upload_date, video_id
                ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    video_data.get('channel_type', ''),
                    video_data.get('topic', ''),
                    video_data.get('keywords', ''),
                    datetime.now(),
                    video_data.get('video_id', '')
                ))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(colored(f"Error adding video to database: {str(e)}", "red"))
            self.conn.rollback()
            return False
        finally:
            self.disconnect()
    
    def update_video_status(self, video_id, status, message=None):
        """Update video status and log the action"""
        self.connect()
        
        try:
            # Update video status
            self.cursor.execute('''
            UPDATE videos SET status = ? WHERE video_id = ?
            ''', (status, video_id))
            
            # Log the action
            self.cursor.execute('''
            INSERT INTO upload_logs (video_id, timestamp, action, status, message)
            VALUES (?, ?, ?, ?, ?)
            ''', (video_id, datetime.now(), 'status_update', status, message or ''))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(colored(f"Error updating video status: {str(e)}", "red"))
            self.conn.rollback()
            return False
        finally:
            self.disconnect()
    
    def update_performance(self, video_id, metrics):
        """Update performance metrics for a video"""
        self.connect()
        
        try:
            # Check if performance entry exists
            self.cursor.execute('''
            SELECT video_id FROM performance WHERE video_id = ?
            ''', (video_id,))
            
            if self.cursor.fetchone():
                # Update existing entry
                self.cursor.execute('''
                UPDATE performance SET
                    views = ?,
                    likes = ?,
                    comments = ?,
                    shares = ?,
                    watch_time_seconds = ?,
                    ctr = ?,
                    avg_view_duration = ?,
                    last_updated = ?
                WHERE video_id = ?
                ''', (
                    metrics.get('views', 0),
                    metrics.get('likes', 0),
                    metrics.get('comments', 0),
                    metrics.get('shares', 0),
                    metrics.get('watch_time_seconds', 0),
                    metrics.get('ctr', 0),
                    metrics.get('avg_view_duration', 0),
                    datetime.now(),
                    video_id
                ))
            else:
                # Create new entry
                self.cursor.execute('''
                INSERT INTO performance (
                    video_id, views, likes, comments, shares,
                    watch_time_seconds, ctr, avg_view_duration, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    video_id,
                    metrics.get('views', 0),
                    metrics.get('likes', 0),
                    metrics.get('comments', 0),
                    metrics.get('shares', 0),
                    metrics.get('watch_time_seconds', 0),
                    metrics.get('ctr', 0),
                    metrics.get('avg_view_duration', 0),
                    datetime.now()
                ))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(colored(f"Error updating performance metrics: {str(e)}", "red"))
            self.conn.rollback()
            return False
        finally:
            self.disconnect()
    
    def add_to_schedule(self, channel_type, scheduled_time):
        """Add a scheduled upload slot"""
        self.connect()
        
        try:
            self.cursor.execute('''
            INSERT INTO schedule (channel_type, scheduled_time, status)
            VALUES (?, ?, 'pending')
            ''', (channel_type, scheduled_time))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(colored(f"Error adding to schedule: {str(e)}", "red"))
            self.conn.rollback()
            return False
        finally:
            self.disconnect()
    
    def get_upcoming_schedule(self, hours=24):
        """Get upcoming scheduled uploads"""
        self.connect()
        
        try:
            now = datetime.now()
            end_time = now + timedelta(hours=hours)
            
            self.cursor.execute('''
            SELECT id, channel_type, scheduled_time, status, video_id
            FROM schedule
            WHERE scheduled_time BETWEEN ? AND ?
            ORDER BY scheduled_time ASC
            ''', (now, end_time))
            
            results = self.cursor.fetchall()
            schedule = []
            
            for row in results:
                schedule.append({
                    'id': row[0],
                    'channel_type': row[1],
                    'scheduled_time': row[2],
                    'status': row[3],
                    'video_id': row[4]
                })
            
            return schedule
        except Exception as e:
            print(colored(f"Error getting upcoming schedule: {str(e)}", "red"))
            return []
        finally:
            self.disconnect()
    
    def assign_video_to_schedule(self, schedule_id, video_id):
        """Assign a video to a scheduled slot"""
        self.connect()
        
        try:
            self.cursor.execute('''
            UPDATE schedule SET video_id = ?, status = 'assigned'
            WHERE id = ?
            ''', (video_id, schedule_id))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(colored(f"Error assigning video to schedule: {str(e)}", "red"))
            self.conn.rollback()
            return False
        finally:
            self.disconnect()
    
    def update_schedule_status(self, schedule_id, status):
        """Update the status of a scheduled upload"""
        self.connect()
        
        try:
            self.cursor.execute('''
            UPDATE schedule SET status = ?
            WHERE id = ?
            ''', (status, schedule_id))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(colored(f"Error updating schedule status: {str(e)}", "red"))
            self.conn.rollback()
            return False
        finally:
            self.disconnect()
    
    def check_topic_exists(self, channel_type, topic, days=60):
        """Check if a topic has been used recently"""
        self.connect()
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            self.cursor.execute('''
            SELECT COUNT(*) FROM content_history
            WHERE channel_type = ? AND topic LIKE ? AND upload_date > ?
            ''', (channel_type, f"%{topic}%", cutoff_date))
            
            count = self.cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            print(colored(f"Error checking topic existence: {str(e)}", "red"))
            return False
        finally:
            self.disconnect()
    
    def get_top_performing_content(self, channel_type, limit=10):
        """Get top performing content for a channel"""
        self.connect()
        
        try:
            self.cursor.execute('''
            SELECT v.video_id, v.title, v.topic, p.views, p.likes, p.comments, 
                   p.shares, p.watch_time_seconds, p.ctr, p.avg_view_duration
            FROM videos v
            JOIN performance p ON v.video_id = p.video_id
            WHERE v.channel_type = ?
            ORDER BY p.views DESC
            LIMIT ?
            ''', (channel_type, limit))
            
            results = self.cursor.fetchall()
            top_content = []
            
            for row in results:
                top_content.append({
                    'video_id': row[0],
                    'title': row[1],
                    'topic': row[2],
                    'views': row[3],
                    'likes': row[4],
                    'comments': row[5],
                    'shares': row[6],
                    'watch_time_seconds': row[7],
                    'ctr': row[8],
                    'avg_view_duration': row[9]
                })
            
            return top_content
        except Exception as e:
            print(colored(f"Error getting top performing content: {str(e)}", "red"))
            return []
        finally:
            self.disconnect()
    
    def get_performance_report(self, days=7):
        """Generate a performance report for the last X days"""
        self.connect()
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            self.cursor.execute('''
            SELECT v.channel_type, 
                   COUNT(v.id) as video_count,
                   SUM(p.views) as total_views,
                   SUM(p.likes) as total_likes,
                   SUM(p.comments) as total_comments,
                   SUM(p.shares) as total_shares,
                   AVG(p.ctr) as avg_ctr,
                   AVG(p.avg_view_duration) as avg_duration
            FROM videos v
            JOIN performance p ON v.video_id = p.video_id
            WHERE v.upload_date > ?
            GROUP BY v.channel_type
            ''', (cutoff_date,))
            
            results = self.cursor.fetchall()
            report = {}
            
            for row in results:
                report[row[0]] = {
                    'video_count': row[1],
                    'total_views': row[2],
                    'total_likes': row[3],
                    'total_comments': row[4],
                    'total_shares': row[5],
                    'avg_ctr': row[6],
                    'avg_duration': row[7]
                }
            
            return report
        except Exception as e:
            print(colored(f"Error generating performance report: {str(e)}", "red"))
            return {}
        finally:
            self.disconnect()
    
    def get_upload_logs(self, video_id=None, limit=50):
        """Get upload logs for a video or all recent logs"""
        self.connect()
        
        try:
            if video_id:
                self.cursor.execute('''
                SELECT id, video_id, timestamp, action, status, message
                FROM upload_logs
                WHERE video_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                ''', (video_id, limit))
            else:
                self.cursor.execute('''
                SELECT id, video_id, timestamp, action, status, message
                FROM upload_logs
                ORDER BY timestamp DESC
                LIMIT ?
                ''', (limit,))
            
            results = self.cursor.fetchall()
            logs = []
            
            for row in results:
                logs.append({
                    'id': row[0],
                    'video_id': row[1],
                    'timestamp': row[2],
                    'action': row[3],
                    'status': row[4],
                    'message': row[5]
                })
            
            return logs
        except Exception as e:
            print(colored(f"Error getting upload logs: {str(e)}", "red"))
            return []
        finally:
            self.disconnect()
    
    def export_to_csv(self, table, output_path):
        """Export a table to CSV for analysis"""
        self.connect()
        
        try:
            query = f"SELECT * FROM {table}"
            df = pd.read_sql_query(query, self.conn)
            df.to_csv(output_path, index=False)
            print(colored(f"Exported {table} to {output_path}", "green"))
            return True
        except Exception as e:
            print(colored(f"Error exporting to CSV: {str(e)}", "red"))
            return False
        finally:
            self.disconnect() 