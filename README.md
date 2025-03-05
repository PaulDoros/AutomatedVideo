# YouTube Shorts Automation System

A comprehensive system for automating the creation, scheduling, and optimization of YouTube Shorts content across multiple channels.

## Features

### Content Generation
- Automated script generation for multiple channel types
- Video generation with dynamic voice selection
- Thumbnail generation
- Content uniqueness validation to prevent repetition

### Scheduling System
- Intelligent posting schedule based on optimal times for each channel
- Time zone optimization for global audience targeting
- 6 posts per day per channel (configurable)
- Performance-based schedule optimization

### Content Monitoring
- Tracks past uploads to prevent content repetition
- Minimum 60-day gap before reusing topics
- Similarity detection for content and topics
- Alternative topic suggestions when duplicates are detected

### Performance Analysis
- Tracks views, likes, comments, and engagement metrics
- Identifies high-performing content patterns
- Generates detailed performance reports with visualizations
- AI-powered content optimization suggestions

### YouTube API Integration
- Automated uploads with proper metadata
- Scheduled posting
- Error handling and retry logic
- Authentication persistence

## Enhanced Video Variety Feature

The system now uses an intelligent categorized video management system that provides more variety in the generated videos while efficiently managing storage. This feature:

1. **Smart Video Categorization**: Videos are now stored in categories based on their content (e.g., "coffee", "programming", "fitness"), allowing for more relevant video selection across different channels.

2. **Content-Aware Video Selection**: The system analyzes scripts to identify key themes and objects, then selects videos from matching categories for more contextually relevant backgrounds.

3. **Cross-Channel Video Sharing**: Videos can be reused across different channels when the content themes overlap, reducing redundant downloads while maintaining variety.

4. **Intelligent Storage Management**: The system automatically cleans up older videos while preserving newer content, keeping the video library at a manageable size.

5. **On-Demand Video Downloading**: New videos are downloaded only when needed, with search terms derived from script analysis for maximum relevance.

This enhancement ensures that your videos will have more variety and visual interest, with backgrounds that match the content themes, while efficiently managing storage space.

## YouTube Upload Enhancements

The system now features improved YouTube upload capabilities that make your videos more professional and algorithm-friendly:

1. **Smart Title Generation**: The system extracts catchy thumbnail titles from the script generation process and uses them as YouTube video titles, ensuring consistency between thumbnails and video titles.

2. **Professional Descriptions**: Videos are now uploaded with professionally formatted descriptions that include:
   - Channel-specific introductions
   - A preview of the script content
   - Clear calls to action for viewers
   - Strategic hashtags relevant to the content type

3. **Dual Image API Integration**: The thumbnail generator now leverages both Pexels and Pixabay APIs to find the most relevant images for thumbnails, with intelligent fallback mechanisms if one API fails.

4. **Content-Specific Hashtags**: Each channel type has a curated set of hashtags that are automatically included in video descriptions to improve discoverability.

5. **Enhanced Metadata Storage**: All video metadata is properly stored in the database for future reference and analytics.

These enhancements significantly improve the professionalism of your YouTube uploads and help your videos perform better in the YouTube algorithm.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/youtube-shorts-automation.git
cd youtube-shorts-automation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
YOUTUBE_CHANNEL_TECH=your_tech_channel_id
YOUTUBE_CHANNEL_TECH_NAME=your_tech_channel_name
YOUTUBE_CHANNEL_AI=your_ai_channel_id
YOUTUBE_CHANNEL_AI_NAME=your_ai_channel_name
YOUTUBE_CHANNEL_PARENTING=your_parenting_channel_id
YOUTUBE_CHANNEL_PARENTING_NAME=your_parenting_channel_name
YOUTUBE_CHANNEL_MEALS=your_meals_channel_id
YOUTUBE_CHANNEL_MEALS_NAME=your_meals_channel_name
YOUTUBE_CHANNEL_FITNESS=your_fitness_channel_id
YOUTUBE_CHANNEL_FITNESS_NAME=your_fitness_channel_name
PEXELS_API_KEY=your_pexels_api_key
PIXABAY_API_KEY=your_pixabay_api_key
```

4. Set up YouTube API credentials:
- Create a project in the [Google Cloud Console](https://console.cloud.google.com/)
- Enable the YouTube Data API v3
- Create OAuth 2.0 credentials
- Download the credentials JSON file and save it as `Backend/client_secret.json`

## Usage

### Basic Usage

Generate and upload content for all channels:
```bash
python Backend/generate_and_upload.py --generate
```

Generate content for a specific channel:
```bash
python Backend/generate_and_upload.py --generate --channel tech_humor --topic "Funny Coding Mistakes"
```

### Video Library Management

Clean up the video library to prevent excessive accumulation:
```bash
python Backend/generate_and_upload.py --cleanup
```

Customize cleanup parameters:
```bash
python Backend/generate_and_upload.py --cleanup --max-videos 30 --days-to-keep 60
```

### Scheduling

Generate a posting schedule for the next 7 days:
```bash
python Backend/generate_and_upload.py --schedule 7
```

Process scheduled uploads for the next hour:
```bash
python Backend/generate_and_upload.py --process 1
```

### Performance Analysis

Analyze performance of all channels:
```bash
python Backend/generate_and_upload.py --analyze
```

### Content Monitoring

Monitor content across all channels:
```bash
python Backend/generate_and_upload.py --monitor
```

## System Architecture

The system consists of several key components:

1. **Content Generator**: Creates scripts, videos, and thumbnails
2. **Video Scheduler**: Manages posting schedules
3. **Content Monitor**: Tracks past uploads and prevents repetition
4. **Performance Analyzer**: Analyzes video performance
5. **YouTube Uploader**: Handles API integration and uploads

## Database Schema

The system uses SQLite for data storage with the following tables:

- `videos`: Stores video metadata
- `performance`: Tracks performance metrics
- `schedule`: Manages posting schedule
- `content_history`: Tracks past content
- `upload_logs`: Logs upload activities

## Automated Workflow

1. Generate posting schedule for optimal times
2. Generate unique content for each channel
3. Upload videos according to schedule
4. Monitor performance metrics
5. Adjust future content and schedules based on performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT models used in content generation
- Google for YouTube Data API
- Pexels and Pixabay for image and video APIs
- All open-source libraries used in this project