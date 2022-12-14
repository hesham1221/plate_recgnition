import os
import cv2
import easyocr
from flask import Flask, render_template, request, redirect, flash
from sqlalchemy.exc import IntegrityError
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'asrtarstaursdlarsn'
app.config["IMAGE_UPLOADS"] = "./static/images/profiles"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]



def allowed_image(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        image = request.files["img_name"]
        if image.filename == "":
            flash('Please Upload Image file', "danger")
            return redirect(request.url)
        if allowed_image(image.filename):
            faceCascade = cv2.CascadeClassifier(
                'haarcascade_russian_plate_number.xml')
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
            img = cv2.imread(os.path.join(app.config["IMAGE_UPLOADS"], filename))
            cv2.imshow("original", img)
            # img = cv2.resize(img, (100, 100))
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(img, scaleFactor=1.2,
                                                minNeighbors=5, minSize=(25, 25))

            for (x, y, w, h) in faces:
                # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                plate = img[y-40: y+h, x:x+w]
                ocr = img[y: y+h, x:x+w]
                # plate = cv2.blur(plate, ksize=(20, 20))
                # put the blurred plate into the original image
                # img[y: y+h, x:x+w] = plate


            cv2.imshow('plates', plate)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
            # os.environ['TESSDATA_PREFIX'] = '.'
            # img = cv.imread("plate.png")
            # # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            cv2.imshow("img", ocr)

            # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            # print(pytesseract.image_to_string(ocrimg, lang="ara", config="."))

            reader = easyocr.Reader(['ar', 'en'], gpu=False)
            result = reader.readtext(ocr)
            print(result)
            cv2.waitKey(0)

            cv2.destroyAllWindows()
            flash('File upload Successfully !', "success")
            return result
        else:
            flash('That file extension is not allowed', "danger")
            return redirect(request.url)


# run always put in last statement or put after all @app.route
if __name__ == '__main__':
    app.run(host='localhost')