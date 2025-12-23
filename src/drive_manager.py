from google.colab import drive, files
import os

class DriveManager:
    def __init__(self):
        drive.mount('/content/drive', force_remount=True)
        self.base_path = '/content/drive/MyDrive/ALIA_ChatBot'
        os.makedirs(self.base_path, exist_ok=True)
    
    def upload_files(self):
        uploaded = files.upload()
        for filename in uploaded.keys():
            dest = os.path.join(self.base_path, filename)
            with open(dest, 'wb') as f:
                f.write(uploaded[filename])
        return list(uploaded.keys())
    
    def list_pdfs(self):
        return [f for f in os.listdir(self.base_path) if f.endswith('.pdf')]
