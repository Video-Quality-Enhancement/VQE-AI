import json

class VideoEnhanceRequest:
    def __init__(self, userId, requestId, videoUrl):
        self.userId = userId
        self.requestId = requestId
        self.videoUrl = videoUrl
    
    def __str__(self):
        return f"VideoEnhanceRequest({self.__dict__})\n"
    
    @classmethod
    def loads(cls, bytes):
        d = json.loads(bytes)
        return cls(**d)