import dotenv
import os
import datetime
from google.cloud import storage
from google.oauth2.service_account import Credentials

class GoogleCloudStorage:
    def __init__(self) -> None:
        credentials = Credentials.from_service_account_file("credentials.json")
        self.storage_client = storage.Client(credentials=credentials, project=os.getenv("PROJECT_ID"))
        self.bucket = self.storage_client.get_bucket(os.getenv("BUCKET_NAME"))

    def upload(self, filepath: str, emailid: str=None):
        # get filename from filepath
        filename = f"enhanced/{filepath.split('/')[-1]}"

        # upload the file to the bucket
        blob = self.bucket.blob(filename)
        blob.upload_from_filename(filepath)

        print(f"File {filepath} uploaded to {os.getenv('BUCKET_NAME')}")

        # Granting access to the file to the emailid
        # myfile_blob.acl.user(emailid).grant_read()

        # get signed url for the emailid
        # expiration_time = datetime.timedelta(hours=1)
        # url = myfile_blob.generate_signed_url(expiration_time, method="GET")
        # url = myfile_blob._get_download_url

        return f"https://storage.googleapis.com/{os.getenv('BUCKET_NAME')}/{filename}"
    
    def list_files(self):
        blobs = self.bucket.list_blobs()
        for blob in blobs:
            print(blob.name)