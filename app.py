import os
import zipfile
from Main import suicide
import main_single
from flask import Flask, render_template, request, redirect, url_for ,send_file

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html') 

upload_dir = os.getcwd() + "/upload_dir"

@app.route("/upload", methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        if len(os.listdir(upload_dir)) != 0:
            for fname in os.listdir(upload_dir):
                fpath = os.path.join(upload_dir, fname)
                os.remove(fpath)
        upload_file = request.files.get('file', None)
        if upload_file.filename != '':
            upload_file.save(os.path.join(upload_dir, upload_file.filename))
            return redirect(url_for('download'))
    return render_template('loading.html')

@app.route("/input")
def input_page():
    return render_template('upload.html')

@app.route("/form", methods=["GET", "POST"])
def form():
    return render_template('form.html')

@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        dic = {
            "family_size": request.form.get("family"),
            "annual_income": request.form.get("income"),
            "eating_habits": request.form.get("eating"),
            "addiction_friend": request.form.get("friendaddiction"),
            "addiction": request.form.get("addiction"),
            "medical_history": request.form.get("medical_history"),
            "depressed": request.form.get("depressed"),
            "anxiety": request.form.get("anxiety"),
            "happy_currently": request.form.get("happy"),
            # "suicidal_thoughts": request.form.get("suicidal_thoughts")
        }
        print(dic)
        x = main_single.suicide(dic)
        print("this is X \n\n\n")
        print(x)
        print(type(x))
        print(type(x[0]))
        if x[0] != None:
            if int(x[0]) == 0:
                print("\n\n\tI AM NOT SUICIDAL\n\n")
                return render_template('not_suicidal.html')
            else:
                return render_template('suicidal.html')

@app.route("/test")
def test():
    return render_template('not_suicidal.html')


@app.route("/download")
def download():
    suicide()
    return render_template('download.html')

@app.route("/load")
def loading_page():
    return render_template('loading.html')

@app.route("/download_files", methods=['GET', 'POST'])
def download_files():
    zipfolder = zipfile.ZipFile('output.zip', 'w', compression=zipfile.ZIP_STORED)
    for root,dirs, files in os.walk('output_result'):
        for file in files:
            zipfolder.write("output_result/"+file)
        zipfolder.close()
    return send_file('output.zip', as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

