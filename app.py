from flask import Flask, request, render_template
import sqlite3

app = Flask(__name__)

def query_db(sql, args=()):
    conn = sqlite3.connect("events.db")
    c = conn.cursor()
    c.execute(sql, args)
    rows = c.fetchall()
    conn.close()
    return rows

@app.route("/")
def index():
    date = request.args.get("date")
    results = []
    if date:
        results = query_db("SELECT date, event FROM events WHERE date=?", (date,))
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
