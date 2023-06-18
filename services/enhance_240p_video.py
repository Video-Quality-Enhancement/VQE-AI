import time
from models import VideoEnhanceRequest
from models import EnhancedVideoResponse
from enhance import main_240p

def enhance_240p_video(video_enhance_request: VideoEnhanceRequest) -> EnhancedVideoResponse:
    print(video_enhance_request)
    
    try:
        enhanced_video_url, enhanced_video_quality, status, statusMessage = main_240p.main(video_enhance_request.videoUrl, video_enhance_request.requestId)
    except Exception as e:
        enhanced_video_url = None
        enhanced_video_quality = None
        status = "FAILED"
        statusMessage = f"Video enhancement failed due to: {e}"

    # create response object
    enhanced_video_response = EnhancedVideoResponse(video_enhance_request, enhanced_video_url, enhanced_video_quality, status, statusMessage)
    return enhanced_video_response