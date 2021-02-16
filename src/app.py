# mongodb+srv://[username:password@]host[/[database][?options]]
import os
from flask import Flask, render_template, request, url_for
import pickle
from model_images import readImage
from kyc_face import verify_images
from aadhar2 import save_face1
from aadhar3 import save_face2
from PIL import Image
import cv2
import numpy as np
from model_images import cv2_to_pil
import time

__author__ = "Alpha3"

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/home')
def indexVideo():
    ss(camera)
    return render_template("index.html")

@app.route('/aboutUs')
def aboutUs():
    return render_template("aboutus.html")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print(APP_ROOT)
@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, "images/")
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    i = 0
    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, "a"+ str(i) +".png"])
        i += 1
        print(destination)
        file.save(destination)

    save_face1("D:\T\SpitHackathon\src\images/a0.png")
    img_1 = readImage("D:\T\SpitHackathon\src\images/a.jpeg")
    save_face2("D:\T\SpitHackathon\src\data/frame0.jpg")
    img_2 = readImage("D:\T\SpitHackathon\src\images/b.jpeg")
    output = verify_images(img_1, img_2)

    if(output):
        return render_template("complete.html")
    else:
        return render_template("notComplete.html")

camera = cv2.VideoCapture(0)  # use 0 for web camera


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
async def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return await Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video')
def indexV():
    """Video streaming home page."""
    return render_template('video.html')



def ss(camera):

    try:

        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')

        # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = 0

    while (currentframe<6):
        time.sleep(2)  # take schreenshot every 5 seconds
        # reading from frame

        ret, frame = camera.read()
        if ret:
            # if video is still left continue creating images
            name = './data/frame' + str(currentframe) + '.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    camera.release()
    cv2.destroyAllWindows()


if(__name__=='__main__'):
    app.run(port=4555, debug=True)
