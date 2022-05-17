from flask import Flask, render_template, request, redirect, url_for ,send_file

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def what_world():
    return render_template('index.html')


@app.route("/world", methods=["GET", "POST"])
def world():
    if request.method == "POST":
        var_world = request.form.get("whatever")
        print(type(var_world))
        print(var_world)
        return f"<h1>You chose</h1><br><h3>{var_world}</h3>"
    else:
        return "<h1>Nothing Happend</h1"


if __name__ == "__main__":
    app.run(debug=True)
