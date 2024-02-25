from main import *
from SkinDisease import *

from pathlib import Path

from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, StringVar, filedialog

ASSETS_PATH = "assets"

OUTPUT_PATH = Path(__file__).parent


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def displayVideo():
    start_video(pathToVideo.get())

def StartWebcam():
    webcam_detect()

def browseFile():
    filename = filedialog.askopenfilename(title="Select a File",
                                          filetypes=(
                                          ("mp4 files", "*.mp4*"), ("avi files", "*.avi*"), ("all files", "*.*")))
    pathToVideo.set(filename)

def browseFileSkin():
    filename = filedialog.askopenfilename(title="Select a File",
                                          filetypes=(
                                          ("png files", "*.png*"), ("jpg files", "*.jpg*"), ("all files", "*.*")))
    pathToImage.set(filename)

def DetectSkin():
    skinDetection(window, pathToImage.get())

def getStreamUrl():
    streamCam(streamingUrl.get())

def getPassUserCam():
    passCam(username.get(), password.get())

def startIpCamera():
    if not entry_3.get():
        getPassUserCam()

    else:
        getStreamUrl()

def troll():
    print(pathToVideo)

if __name__ == '__main__':
    window = Tk()

    window.geometry("1246x702")
    window.configure(bg = "#3B4982")


    canvas = Canvas(
        window,
        bg = "#3B4982",
        height = 702,
        width = 1246,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )

    canvas.place(x = 0, y = 0)
    canvas.create_rectangle(
        0.0,
        0.0,
        221.0,
        702.0,
        fill="#2C3869",
        outline="")

    canvas.create_rectangle(
        209.0,
        0.0,
        1234.0,
        702.0,
        fill="#3B4982",
        outline="")

    image_image_1 = PhotoImage(
        file=relative_to_assets("image_1.png"))
    image_1 = canvas.create_image(
        505.0,
        200.0,
        image=image_image_1
    )

    # User id
    entry_image_1 = PhotoImage(
        file=relative_to_assets("entry_1.png"))
    entry_bg_1 = canvas.create_image(
        535.5,
        267.0,
        image=entry_image_1
    )
    entry_1 = Entry(
        bd=0,
        bg="#A6A6A6",
        fg="#000716",
        highlightthickness=0
    )
    entry_1.place(
        x=405.0,
        y=253.0,
        width=261.0,
        height=26.0
    )

    canvas.create_text(
        313.0,
        259.0,
        anchor="nw",
        text="User ID:",
        fill="#FFFFFF",
        font=("ArimoRoman Regular", 16 * -1)
    )

    # Access Token
    entry_image_2 = PhotoImage(
        file=relative_to_assets("entry_2.png"))
    entry_bg_2 = canvas.create_image(
        556.5,
        224.0,
        image=entry_image_2
    )
    entry_2 = Entry(
        bd=0,
        bg="#A6A6A6",
        fg="#000716",
        highlightthickness=0
    )
    entry_2.place(
        x=447.0,
        y=210.0,
        width=219.0,
        height=26.0
    )

    canvas.create_text(
        313.0,
        213.0,
        anchor="nw",
        text="Access Token:",
        fill="#FFFFFF",
        font=("ArimoRoman Regular", 16 * -1)
    )

    image_image_2 = PhotoImage(
        file=relative_to_assets("image_2.png"))
    image_2 = canvas.create_image(
        637.0,
        105.0,
        image=image_image_2
    )

    image_image_3 = PhotoImage(
        file=relative_to_assets("image_3.png"))
    image_3 = canvas.create_image(
        505.0,
        520.0,
        image=image_image_3
    )

    image_image_4 = PhotoImage(
        file=relative_to_assets("image_4.png"))
    image_4 = canvas.create_image(
        976.0,
        360.0,
        image=image_image_4
    )

    canvas.create_text(
        404.0,
        97.0,
        anchor="nw",
        text="KET NOI VOI MANG XA HOI",
        fill="#FFFFFF",
        font=("Bungee Regular", 14 * -1)
    )

    canvas.create_text(
        314.0,
        416.0,
        anchor="nw",
        text="PHAT HIEN BENH NGOAI DA",
        fill="#FFFFFF",
        font=("Bungee Regular", 20 * -1)
    )

    canvas.create_text(
        313.0,
        143.0,
        anchor="nw",
        text="ID Telegram: ",
        fill="#FFFFFF",
        font=("ArimoRoman Regular", 16 * -1)
    )

    image_image_5 = PhotoImage(
        file=relative_to_assets("image_5.png"))
    image_5 = canvas.create_image(
        638.0,
        426.0,
        image=image_image_5
    )

    # Ket qua
    entry_image_3 = PhotoImage(
        file=relative_to_assets("entry_3.png"))
    entry_bg_3 = canvas.create_image(
        544.0,
        616.5,
        image=entry_image_3
    )
    entry_3 = Entry(
        bd=0,
        bg="#A6A6A6",
        fg="#000716",
        highlightthickness=0
    )
    entry_3.place(
        x=457.5,
        y=603.0,
        width=173.0,
        height=25.0
    )

    # Id telegram
    entry_image_4 = PhotoImage(
        file=relative_to_assets("entry_4.png"))
    entry_bg_4 = canvas.create_image(
        549.5,
        155.0,
        image=entry_image_4
    )
    entry_4 = Entry(
        bd=0,
        bg="#A6A6A6",
        fg="#000716",
        highlightthickness=0
    )
    entry_4.place(
        x=440.0,
        y=141.0,
        width=219.0,
        height=26.0
    )

    image_image_6 = PhotoImage(
        file=relative_to_assets("image_6.png"))
    image_6 = canvas.create_image(
        110.0,
        94.0,
        image=image_image_6
    )

    button_image_1 = PhotoImage(
        file=relative_to_assets("button_1.png"))
    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: print("button_1 clicked"),
        relief="flat"
    )
    button_1.place(
        x=470.0,
        y=292.0,
        width=70.0,
        height=28.0
    )

    canvas.create_text(
        390.0,
        471.0,
        anchor="nw",
        text="Đường dẫn file ảnh",
        fill="#FFFFFF",
        font=("ArimoRoman Regular", 16 * -1)
    )

    # Duong dan da
    pathToImage = StringVar()
    entry_image_5 = PhotoImage(
        file=relative_to_assets("entry_5.png"))
    entry_bg_5 = canvas.create_image(
        454.5,
        512.5,
        image=entry_image_5
    )
    entry_5 = Entry(
        bd=0,
        bg="#A6A6A6",
        textvariable=pathToImage,
        fg="#000716",
        highlightthickness=0
    )
    entry_5.place(
        x=338.5,
        y=499.0,
        width=232.0,
        height=25.0
    )

    # Chon file da
    button_image_2 = PhotoImage(
        file=relative_to_assets("button_2.png"))
    button_2 = Button(
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=browseFileSkin,
        relief="flat"
    )
    button_2.place(
        x=599.0,
        y=499.0,
        width=86.0,
        height=28.0
    )

    # Bat dau da
    button_image_3 = PhotoImage(
        file=relative_to_assets("button_3.png"))
    button_3 = Button(
        image=button_image_3,
        borderwidth=0,
        highlightthickness=0,
        command=DetectSkin,
        relief="flat"
    )
    button_3.place(
        x=464.0,
        y=544.0,
        width=86.0,
        height=28.0
    )

    button_image_4 = PhotoImage(
        file=relative_to_assets("button_4.png"))
    button_4 = Button(
        image=button_image_4,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: print("button_4 clicked"),
        relief="flat"
    )
    button_4.place(
        x=44.0,
        y=215.0,
        width=133.0,
        height=49.0
    )

    button_image_5 = PhotoImage(
        file=relative_to_assets("button_5.png"))
    button_5 = Button(
        image=button_image_5,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: print("button_5 clicked"),
        relief="flat"
    )
    button_5.place(
        x=44.0,
        y=301.0,
        width=133.0,
        height=57.0
    )

    button_image_6 = PhotoImage(
        file=relative_to_assets("button_6.png"))
    button_6 = Button(
        image=button_image_6,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: print("button_6 clicked"),
        relief="flat"
    )
    button_6.place(
        x=44.0,
        y=394.0,
        width=133.0,
        height=57.0
    )

    canvas.create_text(
        464.0,
        178.0,
        anchor="nw",
        text="Facebook",
        fill="#FFFFFF",
        font=("Noto Sans", 18 * -1)
    )

    canvas.create_text(
        359.0,
        607.0,
        anchor="nw",
        text="Kết quả:",
        fill="#FFFFFF",
        font=("ArimoRoman Regular", 16 * -1)
    )

    # Dung Ghi am
    button_image_7 = PhotoImage(
        file=relative_to_assets("button_7.png"))
    button_7 = Button(
        image=button_image_7,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: print("button_7 clicked"),
        relief="flat"
    )
    button_7.place(
        x=991.0,
        y=604.0,
        width=86.0,
        height=28.0
    )

    canvas.create_text(
        953.0,
        564.0,
        anchor="nw",
        text="Ghi âm",
        fill="#FFFFFF",
        font=("Noto Sans", 18 * -1)
    )

    # Khoi dong ghi am
    button_image_8 = PhotoImage(
        file=relative_to_assets("button_8.png"))
    button_8 = Button(
        image=button_image_8,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: print("button_8 clicked"),
        relief="flat"
    )
    button_8.place(
        x=901.0,
        y=603.0,
        width=86.0,
        height=28.0
    )

    canvas.create_text(
        921.0,
        86.0,
        anchor="nw",
        text="Nhan dien",
        fill="#FFFFFF",
        font=("Bungee Regular", 20 * -1)
    )

    canvas.create_text(
        958.0,
        147.0,
        anchor="nw",
        text="Video",
        fill="#FFFFFF",
        font=("Noto Sans", 18 * -1)
    )

    canvas.create_text(
        928.0,
        298.0,
        anchor="nw",
        text="IP CAMERA",
        fill="#FFFFFF",
        font=("Noto Sans", 18 * -1)
    )

    canvas.create_text(
        937.0,
        478.0,
        anchor="nw",
        text="Webcam",
        fill="#FFFFFF",
        font=("Noto Sans", 18 * -1)
    )

    canvas.create_text(
        857.0,
        177.0,
        anchor="nw",
        text="Đường dẫn video",
        fill="#FFFFFF",
        font=("ArimoRoman Regular", 16 * -1)
    )

    canvas.create_text(
        824.0,
        338.0,
        anchor="nw",
        text="Streaming URL",
        fill="#FFFFFF",
        font=("ArimoRoman Regular", 16 * -1)
    )

    canvas.create_text(
        1037.0,
        342.0,
        anchor="nw",
        text="Username",
        fill="#FFFFFF",
        font=("ArimoRoman Regular", 16 * -1)
    )

    canvas.create_text(
        1040.0,
        415.0,
        anchor="nw",
        text="Password",
        fill="#FFFFFF",
        font=("ArimoRoman Regular", 16 * -1)
    )

    image_image_7 = PhotoImage(
        file=relative_to_assets("image_7.png"))
    image_7 = canvas.create_image(
        1087.0,
        108.0,
        image=image_image_7
    )

    # Duong dan video
    pathToVideo = StringVar()
    entry_image_6 = PhotoImage(
        file=relative_to_assets("entry_6.png"))
    entry_bg_6 = canvas.create_image(
        921.5,
        218.5,
        image=entry_image_6
    )
    entry_6 = Entry(
        bd=0,
        bg="#A6A6A6",
        textvariable=pathToVideo,
        fg="#000716",
        highlightthickness=0
    )
    entry_6.place(
        x=805.5,
        y=205.0,
        width=232.0,
        height=25.0
    )

    # Streaming Url
    streamingUrl = StringVar()
    entry_image_7 = PhotoImage(
        file=relative_to_assets("entry_7.png"))
    entry_bg_7 = canvas.create_image(
        881.5,
        387.5,
        image=entry_image_7
    )
    entry_7 = Entry(
        bd=0,
        bg="#A6A6A6",
        textvariable=streamingUrl,
        fg="#000716",
        highlightthickness=0
    )
    entry_7.place(
        x=803.5,
        y=374.0,
        width=156.0,
        height=25.0
    )

    # User name
    username = StringVar()
    entry_image_8 = PhotoImage(
        file=relative_to_assets("entry_8.png"))
    entry_bg_8 = canvas.create_image(
        1075.5,
        387.5,
        image=entry_image_8
    )
    entry_8 = Entry(
        bd=0,
        bg="#A6A6A6",
        textvariable=username,
        fg="#000716",
        highlightthickness=0
    )
    entry_8.place(
        x=997.5,
        y=374.0,
        width=156.0,
        height=25.0
    )

    # Password
    password = StringVar()
    entry_image_9 = PhotoImage(
        file=relative_to_assets("entry_9.png"))
    entry_bg_9 = canvas.create_image(
        1075.5,
        455.5,
        image=entry_image_9
    )
    entry_9 = Entry(
        bd=0,
        bg="#A6A6A6",
        textvariable=password,
        fg="#000716",
        highlightthickness=0
    )
    entry_9.place(
        x=997.5,
        y=442.0,
        width=156.0,
        height=25.0
    )

    # Chon file
    button_image_9 = PhotoImage(
        file=relative_to_assets("button_9.png"))
    button_9 = Button(
        image=button_image_9,
        borderwidth=0,
        highlightthickness=0,
        command=browseFile,
        relief="flat"
    )
    button_9.place(
        x=1066.0,
        y=205.0,
        width=86.0,
        height=28.0
    )

    # Bat dau video
    button_image_10 = PhotoImage(
        file=relative_to_assets("button_10.png"))
    button_10 = Button(
        image=button_image_10,
        borderwidth=0,
        highlightthickness=0,
        command=displayVideo,
        relief="flat"
    )
    button_10.place(
        x=931.0,
        y=250.0,
        width=86.0,
        height=28.0
    )

    # Bat dau webcam
    button_image_11 = PhotoImage(
        file=relative_to_assets("button_11.png"))
    button_11 = Button(
        image=button_image_11,
        borderwidth=0,
        highlightthickness=0,
        command=StartWebcam,
        relief="flat"
    )
    button_11.place(
        x=940.0,
        y=514.0,
        width=86.0,
        height=28.0
    )

    # Bat dau ip camera
    button_image_12 = PhotoImage(
        file=relative_to_assets("button_12.png"))
    button_12 = Button(
        image=button_image_12,
        borderwidth=0,
        highlightthickness=0,
        command=startIpCamera,
        relief="flat"
    )
    button_12.place(
        x=836.0,
        y=439.0,
        width=86.0,
        height=28.0
    )
    window.resizable(False, False)
    window.mainloop()
