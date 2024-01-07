###############################################################################
# importing libraries
from torch import autocast
import torch
import requests
from PIL import Image
from PIL import ImageOps
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import ImageOps

# load the pipeline
device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    # variant='fp16',
    torch_dtype=torch.float16,
    use_auth_token=True
).to(device)
""" 
# for control of cuda is available! if not -> Search/GPU -> Hardware Accelerated GPU Configuration -> ON
import torch
print(torch.cuda.is_available())
"""


# url2 = "https://akn-coco.a-cdn.akinoncloud.com/products/2023/05/18/155842/4da71c2c-b327-4817-b0af-3773d0ac4d80_size690x862_cropCenter.jpg"
# url2 = "C:/Users/ASUS/Desktop/kahve.png"
# prompt4 = "Chocolate coffee, Taken under natural light, a photograph can create a warm and inviting atmosphere. If the shoot is outdoors, the sunlight can beautifully highlight the froth on the coffee, adding a lovely emphasis to the cup."
# user_color = "#0000ff"

########################## TASK 1 ##########################

def user_color_adder(prompt, user_color):
    """
    color adder function according to user_color & prompt
    :param prompt: text -> string
    :param user_color: color(hex code) -> string
    :return: text -> string
    """
    text = prompt + "with a specified color hex code: (" + user_color + ") , COLOR : "
    return text


def create_image(init_image, prompt, user_color):
    """
    creating image using prompt & init image (user's selected)
    :param init_image: Image
    :param prompt: user_prompt -> string
    :param user_color: color -> string (hex code)
    :return: images -> Image
    """
    # for url
    """response = requests.get(url)  # take photo
    init_image = Image.open(BytesIO(response.content)).convert("RGB")  # not RGBA"""
    # resizing
    init_image = init_image.resize((512, 512))  # square / can be (768,512)
    user_color_adder(prompt, user_color)  # that func add color to prompt
    prompt_last = user_color_adder(prompt, user_color)  # added color palette
    with autocast("cuda"):
        images = pipe(prompt=prompt_last, image=init_image, strength=0.75, guidance_scale=7.5)  # generating images
    return images  # return image

# taking input from user
user_image = input("Resim dosyasının yolunu girin (örneğin: image.png): ")  # C:/Users/ASUS/Desktop/kahve.png, C:/Users/ASUS/Desktop/kahve2.png
user_prompt = input(
    "Promptu girin: ")  # Chocolate coffee, Taken under natural light, a photograph can create a warm and inviting atmosphere. If the shoot is outdoors, the sunlight can beautifully highlight the froth on the coffee, adding a lovely emphasis to the cup.
user_color = input("Görselde kullanmak için bir renk girin (hex kodu): ")  # #0000ff

# implementation of task1
def task1_imp(user_image, user_prompt, user_color):
    """
    Processes Task 1 by adding color to the user prompt text on the given image.

    :param user_image: image (from task1)
    :param user_prompt: string -> text (user_prompt)
    :param user_color: string -> color (hex code)
    :return: Image -> image & save selected path
    """
    # open with image to user_photo
    user_image = Image.open(user_image)
    # color adder func implementation
    user_color_adder(user_prompt, user_color)
    # generation of image
    images = create_image(user_image, user_prompt, user_color)
    # saving
    if images and hasattr(images, 'images') and isinstance(images.images, list) and isinstance(images.images[0],
                                                                                               Image.Image):

        sonuc_path = "C:/Users/ASUS/PycharmProjects/spaceship-titanic/task1_generated_photo/task1_7.png"
        images.images[0].save(sonuc_path)
        return images
    else:
        print("Error: The images object is empty or does not contain a PIL Image object.")


# applying task 1 implementation & saving generated image
images = task1_imp(user_image, user_prompt, user_color)


########################## TASK 2 ##########################
from PIL import Image, ImageDraw, ImageFont


# import svgwrite

def cerceve(images):
    """
    Adds a white frame around the generated image
    :param images: PIL.Image -> generated image
    :return: PIL.Image -> with added frame
    """
    cerceve_genislik = 350
    # Çerçeve rengini belirle (255 beyaz için)
    cerceve_rengi = (255, 255, 255)
    # Yeni bir görsel oluştur
    yeni_genislik = images.images[0].width + 2 * cerceve_genislik
    yeni_yukseklik = images.images[0].height + 2 * cerceve_genislik + 100
    yeni_gorsel = Image.new("RGB", (yeni_genislik, yeni_yukseklik), cerceve_rengi)
    # Ana görseli yeni görselin içine kopyala
    yeni_gorsel.paste(images.images[0], (cerceve_genislik, cerceve_genislik))
    return yeni_gorsel


def logo_add(yeni_gorsel, logo_path="C:/Users/ASUS/PycharmProjects/spaceship-titanic/logos/cland.png"):
    """
    Adds the specified logo on top of the generated image.
    :param yeni_gorsel: Image -> Image to add the logo to
    :param logo_path: String -> Path of the logo file you want to add.
    :return: PIL.Image.Image -> Image with added logo.
    """
    logo = Image.open(logo_path)
    # Fotoğrafın boyutlarını al
    foto_genislik, foto_yukseklik = yeni_gorsel.size
    # Logo boyutunu belirle
    logo_boyut = (230, 192)  # İhtiyaca göre değiştirilebilir
    kucuk_logo = logo.resize(logo_boyut)
    # Logo'yu ortalanmış konuma yerleştir
    x_konum = (foto_genislik - logo_boyut[0]) // 2
    y_konum = 50  # (foto_yukseklik - logo_boyut[1]) // 4  # Yukarı ortaya daha yakın bir konum
    yeni_gorsel.paste(kucuk_logo, (x_konum, y_konum))
    return yeni_gorsel


def punchline_add(yeni_gorsel, yazi_metni="AI ad banners lead to higher\nconversions ratesxxxx", yazi_rengi=(0, 0, 0)):
    """
    Adds the punchline text below the generated image.
    :param yeni_gorsel: (PIL.Image.Image) -> Image to add the text to.
    :param yazi_metni: string -> The text you want to add.
    :param yazi_rengi: string -> hex code & Text color, default is black (0, 0, 0).
    :return: PIL.Image.Image -> Image with added text.
    """
    # Yazı tipini ve boyutunu belirle
    yazi_tipi = ImageFont.truetype(
        "C:/Users/ASUS/PycharmProjects/spaceship-titanic/fonts/PlayfairDisplay-VariableFont_wght.ttf", size=70)
    print(yazi_metni)
    # Yazının konumunu belirle
    gorsel_genislik, gorsel_yukseklik = yeni_gorsel.size
    # yazi_tipi.getsize(yazi_metni)
    yazi_tipi.size
    x_konum = (yeni_gorsel.width - yazi_tipi.size) // 2
    y_konum = yeni_gorsel.height - 400  # Fotoğrafın alt kısmına 50 piksel mesafe
    # Bir çizim nesnesi oluştur
    draw = ImageDraw.Draw(yeni_gorsel)
    for row in yazi_metni.split("\n"):
        yazi_genislik = draw.textlength(text=row, font=yazi_tipi)  # text=yazi_metni
        # Yazıyı çiz
        bb_l, bb_t, bb_r, bb_b = draw.textbbox((x_konum, y_konum), row)
        x = x_konum + (bb_r - bb_l) / 2
        y = y_konum + (bb_b - bb_t) / 2
        draw.text(((yeni_gorsel.width - yazi_genislik) // 2, int(y)), row, font=yazi_tipi, fill=yazi_rengi)
        yazi_yukseklik = 75
        y_konum += yazi_yukseklik
        """
        draw.text(((yeni_gorsel.width - yazi_genislik) // 2, y_konum), row, font=yazi_tipi, fill=yazi_rengi)
        yazi_yukseklik = draw.textsize(text=row, font=yazi_tipi) # textsize is deprecated
        y_konum += yazi_yukseklik"""
    return yeni_gorsel


def add_button(yeni_gorsel, button_text, button_color="#316346", padding=10, corner_radius=20):
    """
    Adds a button the generated image
    :param yeni_gorsel: PIL.Image.Image
    :param button_text: string -> user button text
    :param button_color: string -> hex code
    :param padding: int -> default is 10
    :param corner_radius: int -> default is 20
    :return: return: PIL.Image.Image -> Image with added button.
    """
    # Görselin genişliği ve yüksekliğini al
    width, height = yeni_gorsel.size
    # Button boyutları ve konumu belirle
    button_width = 400
    button_height = 100
    button_x = (width - button_width) // 2
    button_y = height - 150
    # Bir çizim nesnesi oluştur
    draw = ImageDraw.Draw(yeni_gorsel)

    # Adjust the button dimensions for padding
    button_x += padding
    button_y += padding
    button_width -= 2 * padding
    button_height -= 2 * padding

    # Draw rounded rectangle for the button
    draw.rounded_rectangle(
        [button_x, button_y, button_x + button_width, button_y + button_height],
        fill=button_color,
        radius=corner_radius
    )

    # Buton üzerine metni çiz
    button_font_size = 30
    button_font = ImageFont.truetype("arial.ttf", size=button_font_size)

    # Metni ortalamak için textbbox fonksiyonunu kullanma
    bb_l, bb_t, bb_r, bb_b = draw.textbbox((button_x, button_y, button_x + button_width, button_y + button_height),
                                           button_text, font=button_font)

    # Yazının boyutlarını kontrol et
    text_width = bb_r - bb_l
    text_height = bb_b - bb_t

    # Yazıyı buton sınırları içinde sığdır
    max_text_width = button_width - 2 * padding
    max_text_height = button_height - 2 * padding

    if text_width > max_text_width:
        # Yazının genişliği buton genişliğini aşıyorsa, boyutunu küçült
        button_font_size = int(button_font_size * max_text_width / text_width)
        button_font = ImageFont.truetype("arial.ttf", size=button_font_size)

    if text_height > max_text_height:
        # Yazının yüksekliği buton yüksekliğini aşıyorsa, boyutunu küçült
        button_font_size = int(button_font_size * max_text_height / text_height)
        button_font = ImageFont.truetype("arial.ttf", size=button_font_size)

    # Metni ortalamak için textbbox fonksiyonunu kullanma
    bb_l, bb_t, bb_r, bb_b = draw.textbbox((button_x, button_y, button_x + button_width, button_y + button_height),
                                           button_text, font=button_font)

    text_x = button_x + (button_width - text_width) // 2
    text_y = button_y + (button_height - text_height) // 2
    draw.text((text_x, text_y), button_text, font=button_font, fill="white")

    return yeni_gorsel


def task2_imp(image, logo_path, button_color, punchline_text, button_text):
    """
    Processes the images for Task 2 by adding a frame, logo, punchline, and a button.
    :param image: (PIL.Image.Image) ->  The input image(s) for processing.
    :param logo_path: string -> The logo path that user's selected
    :param button_color: string -> hex code
    :param punchline_text: string -> The text for the punchline.
    :param button_text: string -> The text for the button
    :return:
    """
    if images and hasattr(images, 'images') and isinstance(images.images, list) and isinstance(images.images[0],
                                                                                               Image.Image):
        yeni_gorsel = cerceve(image)
        # logo
        yeni_gorsel = logo_add(yeni_gorsel, logo_path)
        # punchline
        yeni_gorsel = punchline_add(yeni_gorsel=yeni_gorsel, yazi_metni=punchline_text, yazi_rengi=button_color) # yazi_metni="AI ad banners lead to higher\nconversions ratesxxxx"
        # button
        yeni_gorsel = add_button(yeni_gorsel, button_text=button_text, button_color=button_color)
        # Sonucu kaydet
        sonuc_path = "C:/Users/ASUS/PycharmProjects/spaceship-titanic/task2_generated_photo/task2_7.png"
        yeni_gorsel.save(sonuc_path)
        return yeni_gorsel
    else:
        print("Error: The images object is empty or does not contain a PIL Image object.")


# taking input from user
logo_input = input("Logo dosyasının yolunu girin (örneğin: logo.png): ")  # C:\Users\ASUS\PycharmProjects\spaceship-titanic\logos\cland.png
button_color_input = input("Button & Punchline için bir renk girin (hex kodu): ")  # #1411F1 , org color : #316346
punchline_text_input = input("Punchline için bir text girin (Alt satıra geçmek için '\\n' kullanabilirsiniz): ")  # AI ad banners lead to higher\nconversions ratesxxxx  ## https://stackoverflow.com/questions/38401450/n-in-strings-not-working
punchline_text_input = punchline_text_input.replace("\\n", "\n")
button_text_input = input("Button için bir text girin: ")  # Call to action text here! >

new_image = task2_imp(image=images, logo_path=logo_input, button_color=button_color_input, punchline_text=str(punchline_text_input),
          button_text=button_text_input)


########################## TASK 3 ##########################

# app.py