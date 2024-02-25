from roboflow import Roboflow
import tkinter as tk
import telepot
from PIL import ImageTk, Image
from openai import OpenAI

rf = Roboflow(api_key="E6qJlVbCEgl643lfSLFR")
project = rf.workspace().project("skin-diseases-detection-ubsjb")
model = project.version(1).model

token = '6591943273:AAGp1NNT_GK3RgJ9E81XXP1zpUAaDYQREA4'
receiver_id = 5606318609

bot = telepot.Bot(token)

disease_translations = {
        "": "Khong xac dinh",
        "1": "Khối u ác tính",
        "Acne": "Mụn",
        "Atopic Dermatitis": "Viêm da dị ứng",
        "Chicken Skin": "Da gà",
        "Eczema": "Bệnh chàm",
        "Eruptive Xanthoma": "U vàng xanthoma",
        "Hansen`s Disease/Leprosy": "Bệnh phong",
        "Hansen`s Disease/Leprosy- severe": "Bệnh phong",
        "Healthy Skin": "Bình thường",
        "Leprosy": "Bệnh phong",
        "Leukocytoclastic Vasculitis": "Viêm mạch bạch cầu",
        "Psoriasis": "Bệnh vẩy nến",
        "Purpura": "Ban xuất huyết",
        "Ringworm": "Nấm ngoài da",
        "Spider Angioma": "U mạch nhện",
        "Vitiligo": "Bệnh bạch biến",
        "Warts": "Mụn Cóc",
        "Xanthelasma": "U vàng quanh mắt",
    }

client = OpenAI(
    api_key="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",
)

def skinDetection(root, path):
    detected_diseases = []

    preds = model.predict(path, confidence=40, overlap=30).json()
    detections = preds['predictions']
    for detect in detections:
        classes = detect['class']
        detected_diseases.append(disease_translations[classes])
        print(disease_translations[classes])

    print(path)
    imgDi = Image.open(path)
    imgDi = imgDi.resize((300, 300))
    imgDi = ImageTk.PhotoImage(imgDi)

    display_window = tk.Toplevel(root)
    display_window.title("Detected Skin Diseases")

    image_label_di = tk.Label(display_window, image=imgDi)
    image_label_di.pack()

    disease_label = tk.Label(display_window, text="Bệnh phát hiện được:")
    disease_label.pack()
    diseaseText = ""
    for disease in detected_diseases:
        disease_line = tk.Label(display_window, text=disease)
        diseaseText = diseaseText + disease + ' '
        disease_line.pack()

    display_window.image = imgDi
    bot.sendMessage(receiver_id, 'Các bệnh về da người dùng đang mắc phải: ' + diseaseText)
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Tôi đang mắc các bệnh về da sau đây: " + str(diseaseText) + " hãy cho tôi cách chữa trị"
            },
        ],
    )
    reply = chat.choices[0].message.content
    bot.sendMessage(receiver_id, 'Các bệnh về da người dùng đang mắc phải: ' + reply)


# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())