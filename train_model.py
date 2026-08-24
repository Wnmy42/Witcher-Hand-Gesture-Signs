import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# --- DOSYA YOLLARI (İstediğiniz klasöre göre değiştirebilirsiniz) ---
# Örneğin her şeyi Masaüstündeki bir klasörde toplamak en kolayıdır:
# Cihazınıza göre kullanıcı adını güncelleyebilirsiniz.
klasor_yolu = r"C:\Users\BURAK\Desktop\witsign"

dataset_yolu = os.path.join(klasor_yolu, "dataset.csv")
model_yolu = os.path.join(klasor_yolu, "witcher_model.p")

print(f"Okunan Veri Seti: {dataset_yolu}")

# 1. Veri setini oku
try:
  df = pd.read_csv(dataset_yolu, header=None)
except FileNotFoundError:
  print(f"HATA: '{dataset_yolu}' bulunamadı! Önce veri toplamalısınız.")
  exit()

# 2. Özellikler ve Etiketler olarak ayır
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"Toplam örnek sayısı: {len(X)}")
print(f"Tespit edilen etiketler/büyüler: {np.unique(y)}")

# 3. Eğitim ve test olarak böl
if len(X) > 5:
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  model = KNeighborsClassifier(n_neighbors=3)
  model.fit(X_train, y_train)
  dogruluk = model.score(X_test, y_test)
  print(f"Model Başarı Oranı (Accuracy): %{dogruluk * 100:.2f}")
else:
  model = KNeighborsClassifier(n_neighbors=1)
  model.fit(X, y)
  print("Uyarı: Örnek sayısı az olduğu için test verisi ayrılmadı.")

# 4. Modeli tam yola kaydet
with open(model_yolu, "wb") as f:
  pickle.dump(model, f)

print(f"Model başarıyla kaydedildi -> {model_yolu}")