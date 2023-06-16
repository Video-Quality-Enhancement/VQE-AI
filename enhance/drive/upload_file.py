from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from googleapiclient.http import MediaFileUpload
import os

def get_drive_service():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service


def upload_file(filepath: str):
    file_metadata = {
            'name': os.path.split(filepath)[1],
            'parents': [os.getenv("OUTPUT_FOLDER_ID")]
        }
    drive = get_drive_service()
    media = MediaFileUpload(filepath,
                            mimetype='video/mp4',
                            resumable=True)
    
    print("Uploading...")

    # file = drive.files().create(body=file_metadata,
    #                             media_body=media,
    #                             fields='webViewLink').execute()
    
    file = drive.files().create(body=file_metadata,
                                media_body=media,
                                fields='id').execute()
    
    # enhanced_video_url = file.get('webViewLink')
    enhanced_video_url = f"https://drive.google.com/uc?id={file.get('id')}"
    print(f'Uploaded File With url: {enhanced_video_url}')

    return enhanced_video_url
    

# def main():
#     print_mime_type()
#     update_weights_file()

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    upload_file("enhance/output/output-2/enhanced_video.mp4")