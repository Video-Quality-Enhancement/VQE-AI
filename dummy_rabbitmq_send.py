from models import VideoEnhanceRequest
import json

from pika.adapters.blocking_connection import BlockingChannel
from models import EnhancedVideoResponse
from config import AMQPconnection
import dotenv

dotenv.load_dotenv()

url = "https://rr1---sn-i5uif5t-cags.googlevideo.com/videoplayback?expire=1685472097&ei=Ae91ZKG7KJfrNp78nPAG&ip=212.102.58.178&id=o-ACBDp5xKDtiyoVFB57OOgZYus4cxaIbnRZkAZrjJ2prX&itag=18&source=youtube&requiressl=yes&spc=qEK7B6VdIIkJEQK-JYtRtvoVby0Y22iO4RfOcAm6UQ&vprv=1&svpuc=1&mime=video%2Fmp4&ns=fwocqDolr51n4wbpj_azjDkN&cnr=14&ratebypass=yes&dur=246.665&lmt=1666481094216064&fexp=24007246,24362688,24363391&c=WEB&txp=5538434&n=FpuZfXemKQXgPQ&sparams=expire%2Cei%2Cip%2Cid%2Citag%2Csource%2Crequiressl%2Cspc%2Cvprv%2Csvpuc%2Cmime%2Cns%2Ccnr%2Cratebypass%2Cdur%2Clmt&sig=AOq0QJ8wRgIhAN21lDjZIIidii083_UM-i-xVnwA0QCFPs5n2fNIYrzMAiEAmrQ4OhsbxKF6i-rTAUuNWWmw8VrfLt-SiBJqxXFKalA%3D&redirect_counter=1&rm=sn-vgqeld7s&req_id=daf57ca2277ca3ee&cms_redirect=yes&cmsv=e&ipbypass=yes&mh=ee&mip=115.99.166.93&mm=31&mn=sn-i5uif5t-cags&ms=au&mt=1685450216&mv=m&mvi=1&pl=22&lsparams=ipbypass,mh,mip,mm,mn,ms,mv,mvi,pl&lsig=AG3C_xAwRAIgRYQrgpPsKpWpTWZUpyyJlry9qHsxbevCtehOusVB02QCIHjsUfkYWprr1u1IVA2VmbA8P4bR0O6o8ymqHDH9P_SN"

request = VideoEnhanceRequest(998, 998, url)

request = request.__dict__

connection = AMQPconnection()
producerCh = connection.create_channel()
# result = producerCh.queue_declare(queue="enhanced.video", durable=True)
queue = "360p_queue"

body = json.dumps(request)
producerCh.basic_publish(exchange='', routing_key=queue, body=body)

print("message sent", request)
