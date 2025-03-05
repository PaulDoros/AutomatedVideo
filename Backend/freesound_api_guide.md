# Freesound API Integration Guide

This guide will help you obtain and use a Freesound API key for downloading royalty-free music for your YouTube Shorts videos.

## What is Freesound?

[Freesound](https://freesound.org/) is a collaborative database of Creative Commons Licensed sounds. It allows you to:
- Browse, download and share sounds
- Upload your own sounds
- Access a vast library of audio content through their API

## Getting a Freesound API Key

1. **Create a Freesound Account**:
   - Visit [Freesound.org](https://freesound.org/)
   - Click on "Join now" to create an account
   - Complete the registration process

2. **Apply for an API Key**:
   - Go to [https://freesound.org/apiv2/apply/](https://freesound.org/apiv2/apply/)
   - Log in with your Freesound account
   - Fill out the application form:
     - Provide a name for your application (e.g., "YouTube Shorts Music Provider")
     - Briefly describe how you'll use the API (e.g., "Downloading royalty-free music for YouTube Shorts videos")
     - Accept the terms of service

3. **Receive Your API Key**:
   - After submitting the form, you'll receive your API key via email
   - This process is usually quick (minutes to hours)

## Adding Your API Key to the Project

1. **Update Your .env File**:
   - Open the `.env` file in the root directory of the project
   - Find the line: `FREESOUND_API_KEY=your_freesound_api_key`
   - Replace `your_freesound_api_key` with the actual API key you received

2. **Test Your API Key**:
   - Run the test script to verify your API key is working:
   ```bash
   python Backend/test_freesound.py
   ```
   - If successful, you should see music being downloaded and played

## Using Freesound in Your Videos

The `MusicProvider` class will now automatically use Freesound as the primary source for music when generating videos. The system will:

1. First check for local music files in the channel-specific directories
2. If no local music is found, it will use default music
3. If needed, it will search Freesound for music matching the channel's mood
4. Downloaded music will be saved to the channel directory for future use

## License Considerations

Freesound offers sounds under various Creative Commons licenses. Our integration specifically filters for:
- **Creative Commons 0** (CC0) - These sounds are in the public domain and can be used without attribution

## Troubleshooting

If you encounter issues with the Freesound API:

1. **Check your API key** - Make sure it's correctly entered in the `.env` file
2. **Verify your internet connection** - The API requires internet access
3. **Check API rate limits** - Freesound may have rate limits on API requests
4. **Look for specific error messages** - The test script will display detailed error information

For more information, visit the [Freesound API Documentation](https://freesound.org/docs/api/). 