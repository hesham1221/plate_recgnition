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

def textextract(img):
    reader = easyocr.Reader(['ar', 'en'], gpu=False)
    result = reader.readtext(img)
    return result

def data(color, text):
    plate_data = {}
    exact_text = text[0][1]
    # print("exact", exact_text)
    plate_data["color"] = color
    plate_data["text"] = exact_text
    print(plate_data)
    return plate_data

    # color extract


def colorextract(plate):
    res = cv2.resize(plate, dsize=(250, 200), interpolation=cv2.INTER_CUBIC)
    crop2 = res[0:80, 0:250]
    # reshape array of image pixels
    data = np.reshape(crop2, (80 * 250, 3))
    data = np.float32(data)
    # clustreing using kmeans
    number_clusters = 1  # one dominat color
    # when to stop the algorithm, 10 = max_iter, 1.0 = epsilon
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(
        data, number_clusters, None, criteria, 10, flags)
    rgb_values = []
    for index, row in enumerate(centers):
        rgb = create_bar(200, 200, row)
        # bars.append(bar)
        rgb_values.append(rgb)
    for index, row in enumerate(rgb_values):
        # print(f'RGB{row}')
        return 'red'


def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return (red, green, blue)


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
            # bars = []
            # cv2.imshow('Dominant colors', img_bar)
            flash('File upload Successfully !', "success")
            faceCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
            faces = faceCascade.detectMultiScale(img, scaleFactor=1.2,minNeighbors=5, minSize=(25, 25))
            print(faceCascade)
            for (x, y, w, h) in faces:
                plate = img[y-40: y+h, x:x+w]  # all
                ocr = img[y: y+h, x:x+w]  # cropped
            cv2.imshow('plates', plate)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            text_extraction = textextract(ocr)
            color = colorextract(plate)
            # print(f"text : {text_extraction} , color : {color}")
            palte_info = data(color, text_extraction)
            os.remove(os.path.join(app.config["IMAGE_UPLOADS"], filename))
            return palte_info
        else:
            flash('That file extension is not allowed', "danger")
            return redirect(request.url)


# run always put in last statement or put after all @app.route
if __name__ == '__main__':
    app.run(debug=False , host='0.0.0.0')