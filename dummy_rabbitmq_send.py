from models import VideoEnhanceRequest
import json

from pika.adapters.blocking_connection import BlockingChannel
from models import EnhancedVideoResponse
from config import AMQPconnection
import dotenv

dotenv.load_dotenv()

# url = "https://drive.google.com/uc?id=1yczEIQHc14jc0B412eB5JlouGLU4sobJ"
# url = "https://download.pexels.com/vimeo/822411462/pexels-tommy-t-16609645.mp4?fps=29.97&width=426"
# url = "https://drive.google.com/uc?id=1cvaGrZz1KYyzaqFiNw-9OR-ekTf1UE5F"
# url = "https://drive.google.com/uc?id=14ThGh-bVRi9a1jbT8P894TMoeHLepEii"
# url = "https://rr5---sn-vgqsrnzz.googlevideo.com/videoplayback?expire=1686290425&ei=mWuCZPn1IZKikwa_75GICQ&ip=156.146.48.223&id=o-ACk1CJ0YYmUZACoJgTuEcPmHmSOJXTAE0HEgZtNCYYA0&itag=135&aitags=133%2C134%2C135%2C136%2C160%2C242%2C243%2C244%2C247%2C278&source=youtube&requiressl=yes&spc=qEK7B34iDIVU_F90SSXfGp75ricMeC6SBt5WqNyQww&vprv=1&svpuc=1&mime=video%2Fmp4&ns=iZUICg49LuaVH2mvrYAua5cN&gir=yes&clen=1474360&dur=10.560&lmt=1444130439013219&keepalive=yes&fexp=24007246&c=WEB&n=QDvjv15ibtmKZA&sparams=expire%2Cei%2Cip%2Cid%2Caitags%2Csource%2Crequiressl%2Cspc%2Cvprv%2Csvpuc%2Cmime%2Cns%2Cgir%2Cclen%2Cdur%2Clmt&sig=AOq0QJ8wRAIgZAxwlxMyA3JFZqtdezs9FE-z9aGIFjD6GJfUNyZejAYCIEHCkA_DyFbBlhBKNEGXttQWWCMpvSetZdZMILiKWV4-&redirect_counter=1&cm2rm=sn-nx5ze7e&req_id=861f471539e8a3ee&cms_redirect=yes&cmsv=e&mh=BI&mip=27.7.167.249&mm=34&mn=sn-vgqsrnzz&ms=ltu&mt=1686267786&mv=D&mvi=5&pl=0&lsparams=mh,mip,mm,mn,ms,mv,mvi,pl&lsig=AG3C_xAwRQIgfzSl6EwxH-Skmg5pPGmRVLeW3cBCED-hMEc-LLhsBeMCIQChDbRPHTj5KundTwTcNYhP5iKjFF1LAHMTWHOTuuLmtg%3D%3D"
# url = "https://drive.google.com/uc?id=16ZNwBRrw-q6MA1hTFQc1DjBWhUiz8qSy"
# url = "https://drive.google.com/uc?id=1moOXqselYhV2Oa12zqbn7_w0GcL1oFWp"
url = "https://drive.google.com/uc?id=1-nS_vqPHGmvXVgsiSg1uCOsnGTsmkwVQ"

request = VideoEnhanceRequest('user03', 'requestBBBB', url)

request = request.__dict__

connection = AMQPconnection()
producerCh = connection.create_channel()
# result = producerCh.queue_declare(queue="enhanced.video", durable=True)
queue = "240p_queue"

body = json.dumps(request)
producerCh.basic_publish(exchange='', routing_key=queue, body=body)

print("message sent", request)
