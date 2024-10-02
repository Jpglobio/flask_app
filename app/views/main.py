from flask import Blueprint, render_template, g

main = Blueprint('main', __name__)

@main.route('/')
def home():
    db = g.db
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT * FROM students")
    students = cursor.fetchall()

    cursor.execute("SELECT fname FROM students WHERE fname RLIKE BINARY '^[A-Z]' LIMIT 1")
    fname = cursor.fetchone()

    cursor.execute("SELECT DISTINCT(major) FROM students")
    major = cursor.fetchall()

    cursor.execute("""
     SELECT fname,
           CASE
               WHEN amount = 0 THEN "NULL"
               ELSE amount
           END AS scholarship_amount
    FROM students
     """)
    scholar = cursor.fetchall()


    cursor.execute("SELECT id, fname FROM students WHERE scholarship = 0")
    non_scholar = cursor.fetchall()
    cursor.close()

    return render_template('index.html', students=students, fname=fname, major=major, scholar=scholar, non_scholar=non_scholar)
