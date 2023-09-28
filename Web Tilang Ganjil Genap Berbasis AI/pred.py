# Import Library
from flask import (
    Flask,
    json,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename
from datetime import datetime
from ultralytics import YOLO
import os, requests

# Initiate App
flask_app = Flask(__name__)

# Secret Key
sec_key = os.urandom(256)
flask_app.config["SECRET_KEY"] = sec_key

# Set Path Upload & Result File
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/predict"
CROP_FOLDER = "static/predict/crops/license-plate"
flask_app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
flask_app.config["RESULT_FOLDER"] = RESULT_FOLDER
flask_app.config["CROP_FOLDER"] = CROP_FOLDER
folder_upload = flask_app.config["UPLOAD_FOLDER"]
folder_predict = flask_app.config["RESULT_FOLDER"]
folder_crop = flask_app.config["CROP_FOLDER"]

# Allow File Extensions
ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Get current date
curr_date = datetime.now().strftime("%Y%m%d%H%M%S")

# Set OCR API KEY
OCR_API_KEY = "K84174163488957"


# Function Allow File Extenstions
def allowFile(name_file):
    return os.path.splitext(name_file)[1] in ALLOWED_EXTENSIONS


# Function Delete Image
def delImage():
    # Handle Error while Removing File/Folder
    try:
        # Check uploads Folder is exists & Removing File
        if os.path.exists(folder_upload):
            for file in os.listdir(folder_upload):
                file_upload_path = os.path.join(folder_upload, file)
                if os.path.isfile(file_upload_path) and file.lower().endswith(
                    ALLOWED_EXTENSIONS
                ):
                    os.remove(file_upload_path)
        # Check predict Folder is exists & Removing File/Folder in predict Folder
        if os.path.exists(folder_predict):
            for file in os.listdir(folder_predict):
                file_predict_path = os.path.join(folder_predict, file)
                if os.path.isfile(file_predict_path) and file.lower().endswith(
                    ALLOWED_EXTENSIONS
                ):
                    os.remove(file_predict_path)
            for foldername in os.listdir(folder_predict):
                folder_path = os.path.join(folder_predict, foldername)
                if os.path.isdir(folder_path):
                    folder_contents = os.listdir(folder_path)
                    if len(folder_contents) == 1 and os.path.isdir(
                        os.path.join(folder_path, folder_contents[0])
                    ):
                        subfolder_path = os.path.join(folder_path, folder_contents[0])
                        subfolder_contents = os.listdir(subfolder_path)
                        for subfilename in subfolder_contents:
                            subfile_path = os.path.join(subfolder_path, subfilename)
                            if os.path.isfile(subfile_path):
                                os.remove(subfile_path)
                        os.rmdir(subfolder_path)
                    os.rmdir(folder_path)
            os.rmdir(folder_predict)
    except Exception as e:
        return jsonify(e)


# Function Prediction
def predictionImg(img_file):
    # Define Image
    img = img_file
    # Load Model
    model = YOLO("model/lnpr.pt")
    # Prediction
    pred = model.predict(
        source=img,
        save=True,
        save_crop=True,
        save_conf=True,
        stream=False,
        project="static",
    )
    # Check Empty Result Folder
    result_files = os.listdir(folder_predict)
    if len(result_files) != 0:
        msg = "Modeling Success"
    else:
        msg = "Modeling Failed"
    return msg


# Function OCR
def ocr(img_file, api_key):
    payload = {
        "isOverlayRequired": False,
        "apikey": api_key,
        "language": "eng",
        "detectOrientation": True,
        "OCREngine": 2,
    }
    with open(img_file, "rb") as f:
        req = requests.post(
            "https://api.ocr.space/parse/image",
            files={img_file: f},
            data=payload,
        )
    return req.content.decode()


# Route Upload File
@flask_app.route("/")
def index():
    delImage()
    return render_template("index.html")

# Route About Us File
@flask_app.route("/about-us")
def aboutUs():
    return render_template("about-us.html")


# Route Read File
@flask_app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        # Check POST Request has a File Part
        if "file" not in request.files:
            return redirect("/")
        file = request.files["file"]
        # Check Selected File
        if file.filename == "":
            return redirect("/")
        # Check Both is True
        if file and allowFile(file.filename):
            # Getting Request File
            file = request.files["file"]
            # Getting Extension File
            name_file_without_ext = os.path.splitext(os.path.basename(file.filename))[0]
            # Getting Name File
            file_ext = os.path.splitext(file.filename)[1]
            # Secure File
            name_file = secure_filename(name_file_without_ext + curr_date + file_ext)
            # Save File
            file.save(os.path.join(folder_upload, name_file))
            return redirect(url_for("show_img", filename=name_file))


# Route Show Image
@flask_app.route("/show/<filename>")
def show_img(filename):
    return render_template("read.html", filename=filename)


# Route Detect Image
@flask_app.route("/detect", methods=["POST"])
def detect():
    if request.method == "POST":
        # Getting Name File from Form
        file = request.form["file"]
        # Read Image File
        img_path = os.path.join(folder_upload, file)
        msg = predictionImg(img_path)
        if msg == "Modeling Success":
            img_crop = os.path.join(folder_crop, file)
            return render_template(
                "detect.html", image_predicted=img_crop, img_name=file
            )
        else:
            return redirect("/")


# Route Report
@flask_app.route("/report", methods=["POST"])
def report():
    if request.method == "POST":
        # Getting Image Name from Form
        file = request.form["img_name"]
        # Read Image File
        img_path = os.path.join(folder_crop, file)

        json_data = ocr(img_file=img_path, api_key=OCR_API_KEY)
        parsed_data = json.loads(json_data)
        parsed_txt = parsed_data["ParsedResults"][0]["ParsedText"]
        parsed_txt_split = parsed_txt.split(" ")
        result_txt = []
        result_txt.append(parsed_txt_split[0])

        if parsed_txt_split[1].isdigit():
            result_txt.append(int(parsed_txt_split[1]))
        else:
            result_txt.append(parsed_txt_split[1])

        if '\n' in parsed_txt_split[2]:
            split_txt2 = parsed_txt_split[2].split("\n")
            result_txt.append(split_txt2[0])
            result_txt.append(split_txt2[1])
        else:
            result_txt.append(parsed_txt_split[2])

        num_plate = result_txt[1]
        result_txt = result_txt[0] + " " + str(result_txt[1]) + " " + result_txt[2]
        curr_year = datetime.now().year

        if result_txt != "":
            if (curr_year % 2 == 0 and num_plate % 2 == 0) or (curr_year % 2 == 1 and num_plate % 2 == 1):
                return render_template("report.html", txt_report=result_txt, img_report=img_path, tilang_report="YES")
            elif (curr_year % 2 == 0 and num_plate % 2 == 1) or (curr_year % 2 == 1 and num_plate % 2 == 0):
                return render_template("report.html", filename=file, txt_report=result_txt, img_report=img_path, tilang_report="NO")
        else:
            return redirect("/")


# Run App
if __name__ == "__main__":
    flask_app.run()
