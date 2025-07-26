import os
import sys
import argparse  # Komut satırı argümanları için
import numpy as np
import tensorflow as tf
from PIL import Image

# --- Ayarlar (BURALARI KENDİNİZE GÖRE DÜZENLEYİN) ---
MODEL_PATH = 'main_model.h5'
IMG_WIDTH = 256
IMG_HEIGHT = 256
# ----------------------------------------------------

# TensorFlow'un bilgi mesajlarını gizle (İsteğe Bağlı)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def iou_metric(y_true, y_pred, smooth=1):
    """
    Model yüklenirken gerekli olan özel IoU metrik fonksiyonu.
    İçeriği, modeli eğitirken kullandığınızla aynı olmalıdır.
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def load_change_detection_model(model_path):
    """
    Modeli, özel nesne ile ve derlenmemiş halde yükler.
    """
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı -> {model_path}")
        return None
    
    print(f"'{model_path}' model dosyası yükleniyor...")
    try:
        #custom_objects = {'iou_metric': iou_metric}
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model başarıyla yüklendi.")
        return model
    except Exception as e:
        print(f"Model yüklenirken bir hata oluştu: {e}")
        return None

def preprocess_image(image_path, width, height):
    """
    Tek bir görüntüyü yükler, yeniden boyutlandırır ve normalize eder.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((width, height))
        img_array = np.array(img) / 255.0
        return img_array
    except FileNotFoundError:
        print(f"HATA: Görüntü dosyası bulunamadı -> {image_path}")
        return None

def save_result_mask(prediction_tensor, output_path):
    """
    Modelin tahminini işler ve sonuç maskesini dosyaya kaydeder.
    """
    # Boyutları (1, H, W, 1) -> (H, W) şekline getir
    mask = np.squeeze(prediction_tensor)
    
    # Eşikleme (0.5'ten büyük pikselleri değişim (beyaz) olarak kabul et)
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Numpy dizisinden PIL resim nesnesi oluştur
    result_image = Image.fromarray(mask)
    
    # Sonucu kaydet
    result_image.save(output_path)
    print(f"Başarılı! Değişim maskesi şuraya kaydedildi: {output_path}")

def main():
    """
    Ana program akışı.
    """
    # 1. Komut satırı argümanlarını tanımla ve al
    parser = argparse.ArgumentParser(description="İki uydu görüntüsü arasında değişim tespiti yapar.")
    parser.add_argument("before_image", help="'Önce' durumunu gösteren ilk görüntü dosyası.")
    parser.add_argument("after_image", help="'Sonra' durumunu gösteren ikinci görüntü dosyası.")
    parser.add_argument("output_mask", help="Sonuç değişim maskesinin kaydedileceği dosya yolu.")
    args = parser.parse_args()

    # 2. Modeli yükle
    model = load_change_detection_model(MODEL_PATH)
    if model is None:
        sys.exit(1) # Programı hata koduyla sonlandır

    # 3. Görüntüleri işle
    print("Görüntüler işleniyor...")
    before_img = preprocess_image(args.before_image, IMG_WIDTH, IMG_HEIGHT)
    after_img = preprocess_image(args.after_image, IMG_WIDTH, IMG_HEIGHT)

    if before_img is None or after_img is None:
        sys.exit(1)

    # 4. Görüntüleri modelin beklediği formata göre birleştir
    # Genellikle kanallar birleştirilir (Örn: 3 kanallı + 3 kanallı -> 6 kanallı)
    combined_input = np.concatenate([before_img, after_img], axis=-1)
    
    # Model tek bir örnek beklediği için batch boyutunu ekle: (H, W, C) -> (1, H, W, C)
    input_tensor = np.expand_dims(combined_input, axis=0)
    
    # 5. Tahmin yap
    print("Değişim tespiti yapılıyor...")
    prediction = model.predict(input_tensor)
    
    # 6. Sonucu kaydet
    save_result_mask(prediction, args.output_mask)


if __name__ == "__main__":
    main()