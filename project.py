
import os
import subprocess
import whisper
from pyannote.audio import Pipeline
from collections import defaultdict
import requests

def extract_audio_segment(video_file, start_time, duration):
    output_audio_file = "audio_trim.mp3"
    ffmpeg_path = r"C:\ffmeg\ffmpeg.exe" 
    # FFmpeg command to extract audio from the video and convert it to MP3
    ffmpeg_cmd = [
        ffmpeg_path, "-i", video_file,
        "-ss", str(start_time), "-t", str(duration),
        "-c:a", "libmp3lame", "-b:a", "192k", "-y", output_audio_file
    ]
    
    try:
        print(f"Extracting {duration} seconds of audio starting at {start_time}...")
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"Audio extraction successful: {output_audio_file}")
        return output_audio_file
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio segment: {e}")
        raise e

def extract_audio_bits(video_file, start_time, duration, i, j):
    output_audio_file = f"audio_{i}{j}.mp3"  # Unique filename based on speaker (i) and segment (j)
    ffmpeg_path = r"C:\ffmeg\ffmpeg.exe"  # Path to FFmpeg executable

    # FFmpeg command to extract audio from the video and convert it to MP3
    ffmpeg_cmd = [
        ffmpeg_path, "-i", video_file,
        "-ss", str(start_time), "-t", str(duration),
        "-c:a", "libmp3lame", "-b:a", "192k", "-y", output_audio_file
    ]
    
    try:
        print(f"Extracting {duration} seconds of audio starting at {start_time} for Speaker {i}, Segment {j}...")
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"Audio extraction successful: {output_audio_file}")
        return output_audio_file
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio segment: {e}")
        raise e
    

def transcribe_audio(audio_file):
    """
    Transcribe audio to text using the Whisper model.
    """
    # Load the Whisper model
    model = whisper.load_model("base")

    print(f"Transcribing audio from {audio_file}...")

    # Transcribe the audio file
    result = model.transcribe(audio_file)

    print("Transcription completed!")
    return result["text"]

def speaker_diarization(audio_file):
    """
    Perform speaker diarization to distinguish between speakers.
    """
    Token = "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"  # Replace with your actual token

    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1", 
            use_auth_token=Token
        )
        diarization = diarization_pipeline(audio_file)

        speaker_segments = defaultdict(list)
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments[speaker].append((segment.start, segment.end))

        return speaker_segments

    except Exception as e:
        print(f"Error loading the pipeline: {e}")
        return None

def process_speakers(audio_file, speaker_segments):
    """
    Transcribe audio for each speaker's segment and organize the conversation.
    """
    i = 0  # Speaker number
    conversation = []

    for speaker, segments in speaker_segments.items():
        for j, (start, end) in enumerate(segments):
            print(f"Processing Speaker {speaker}, Segment {start}-{end}...")
            segment_audio = extract_audio_bits(audio_file, start, end - start, i, j)  # Pass i and j
            transcript = transcribe_audio(segment_audio)
            conversation.append(f"Speaker {speaker}: {transcript}")
        i += 1  # Increment the speaker number for the next speaker

    return conversation

def ask_ai(conversation):
    """
    Use Hugging Face's Meta-Llama-3-8B-Instruct model to analyze the conversation and generate a detailed report.
    """
    API_KEY = "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"  # Replace with your actual API key
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    # Prepare the detailed prompt
    data = {
        "inputs": f"""
        Context: This is a professional mentoring session where a mentor is guiding a student. The conversation includes a variety of topics such as technical skills development, career planning, problem-solving strategies, and emotional intelligence. The mentor provides specific advice, resources, and feedback to help the student achieve their goals.

        The student's responses highlight their preferences, enthusiasm, and concerns. The conversation needs to be analyzed for the following details:

        1. **Summary of the Conversation:**
           - Highlight the main themes and topics discussed.
           - Summarize the mentor's key advice and the student's focus areas.

        2. **Mentor's Communication Style:**
           - Analyze the tone and approach used by the mentor.
           - Identify whether the communication was supportive, informative, or directive.
           - Highlight any notable techniques used to engage the student.

        3. **Student's Interests and Engagement:**
           - Identify the student's areas of interest based on their responses.
           - Assess their enthusiasm, curiosity, and level of engagement.
           - Analyze any challenges or concerns raised by the student.

        4. **Sentiment Analysis:**
           - Evaluate the emotional tone of the conversation.
           - Highlight positive, neutral, or negative sentiment patterns.

        5. **Suggestions for Improvement:**
           - Provide actionable suggestions for the mentor to enhance communication and engagement.
           - Tailor these suggestions to align with the student’s preferences and goals.

        6. **Insights for Personalized Mentoring:**
           - Recommend ways the mentor can adapt their approach for better alignment with the student’s needs.
           - Suggest potential resources, topics, or exercises to further the student’s development.

        7. **Patterns and Observations:**
           - Highlight any recurring themes or significant patterns in the conversation.
           - Provide insights into how the mentor-student dynamic can be optimized.

        Data: {conversation}
        """,
        "parameters": {
            "max_length": 30000,  # Increased for a longer, more detailed response
            "temperature": 0.65,  # Slightly reduced for more factual and focused output
            "top_p": 0.85,       # Adjusted for controlled randomness
            "top_k": 60          # Increased the sampling pool
        }
    }

    # Send the request to the Hugging Face API
    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        # Get the generated text (response from model)
        generated_text = response.json()[0]["generated_text"]
        return generated_text
    else:
        return f"Error: {response.status_code}"

if __name__ == "__main__":
    video_file = r"samplemeeting.mp4"  # Replace with your video file path

    # Extract audio from video
    audio_file = extract_audio_segment(video_file, 0, 40)  # First 20 seconds of the meeting

    # Perform speaker diarization
    speaker_segments = speaker_diarization(audio_file)
    if not speaker_segments:
        print("Speaker diarization failed.")
        exit()

    # Process and transcribe speaker-specific audio
    conversation = process_speakers(audio_file, speaker_segments)
    print("Conversation:")
    print(conversation)

    # Send conversation to the AI for analysis
    analysis = ask_ai(conversation)
    print("AI Analysis:")
    print(analysis)























