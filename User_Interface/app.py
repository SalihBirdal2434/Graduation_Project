import os
import sys
from flask import Flask, render_template, request

# Proje root'unu sys.path'e ekle (Model_Building, XAI_and_LLMs vb. import icin)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)


@app.route("/")
def index():
    return render_template("arayuz.html")


@app.route("/analist")
def analist():
    isim = request.args.get("isim", "Bilinmiyor")
    index = request.args.get("index", 10, type=int)

    credit_ok = True
    sonuc = "<p>Kredi skoru analizi tamamlandi. Detaylar asagidaki grafiklerde yer almaktadir.</p>"

    return render_template(
        "analist.html",
        isim=isim,
        credit_ok=credit_ok,
        sonuc=sonuc,
    )



@app.route("/musteri")
def musteri():
    isim = request.args.get("isim", "Bilinmiyor")
    index = request.args.get("index", 10, type=int)

    credit_ok = True
    sonuc = "<p>Kredi skoru analizi tamamlandi. Detaylar asagidaki grafikte yer almaktadir.</p>"

    return render_template(
        "musteri.html",
        isim=isim,
        credit_ok=credit_ok,
        sonuc=sonuc,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
