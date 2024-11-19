from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO

app = Flask(__name__, template_folder='templates')


model = tf.keras.models.load_model("C:/Users/omers/OneDrive/Masaüstü/TEZ-2/50epochmodel.h5")

print(model.input_shape)


# Ana sayfa rotası
@app.route("/")
def index():
    return render_template("giris.html")




# Tahmin rotası (görüntüyü sadece göstermek için)
@app.route("/tespit", methods=["POST"])
def predict():
    # Görüntüyü yükle
    file = request.files["file"]
    img = Image.open(file).resize((150, 150))  # Modelin giriş boyutlarına göre yeniden boyutla

    # Görüntüyü model için hazırlama
    img_array = np.array(img) / 255.0  # Normalize et
    img_array = np.expand_dims(img_array, axis=0)  # Batch boyutunu ekle

    # Model ile tahmin yap
    prediction = model.predict(img_array)  # Çıktı: [0.8], [0.2] gibi bir dizi
    cancer_probability = prediction[0][0]  # Çünkü modelin çıktısı bir vektör.
    cancer_percentage = round(cancer_probability * 100, 2)

    
    # Tahmini yüzdeye çevir
    cancer_percentage = round(cancer_probability * 100, 2)
    
        # Sonuç metni ve rengi
    if cancer_probability > 0.5:
        result = "Malignant"  # Kötü Huylu
        color = "text-danger"  # Kırmızı renk
    elif cancer_probability < 0.5:
        result = "Benign"  # İyi Huylu
        color = "text-success"  # Yeşil renk
    else:
        result = "Orta"
        color = "text-primary"  # Mavi renk
    

    # Görüntüyü base64 formatına çevir
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Tahmin sonucunu ve görüntüyü tespit.html sayfasına gönder
    # HTML'ye gönderme
    return render_template("tespit.html", img_data=img_str, tahmin=cancer_percentage, result=result, color=color)
    







# Hakkımızda sayfası
@app.route("/hakkimizda", methods=["GET"])
def hakkimizda():
    return render_template("hakkimizda.html")

# Cilt Kanseri Tespiti sayfası
@app.route("/kanser_nedir", methods=["GET"])
def tespit():
    return render_template("kanser_nedir.html")




# Uygulamanın çalışması için gerekli ayar
if __name__ == "__main__":
    app.run(port=8000, debug=True)
